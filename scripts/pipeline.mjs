import { createRepo, uploadFiles } from '@huggingface/hub';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import https from 'node:https';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const metadataDir = path.join(repoRoot, 'metadata');
const outputDir = path.join(repoRoot, 'output');

const EEE_CONFIGS = [
  'ace',
  'apex-agents',
  'apex-v1',
  'appworld_test_normal',
  'browsecompplus',
  'global-mmlu-lite',
  'helm_capabilities',
  'helm_instruct',
  'helm_lite',
  'helm_mmlu',
  'hfopenllm_v2',
  'livecodebenchpro',
  'reward-bench',
  'swe-bench',
  'tau-bench-2_airline',
  'tau-bench-2_retail',
  'tau-bench-2_telecom',
];

const DEFAULT_CONFIG_BATCH_SIZE = 4;
const DATASET_REPO = { type: 'dataset', name: 'evijit/ev_card_be' };
const CONFIG_VERSION = 1;
const VERSION_SUFFIX_REGEX = /^(.*?)-((?:19|20)\d{6})(?:-(.+))?$/;
const EEE_DATASET_REPO = 'evaleval/EEE_datastore';
const EEE_DATASET_RAW_BASE = `https://huggingface.co/datasets/${EEE_DATASET_REPO}/raw/main`;
const EEE_DATASET_TREE_API_BASE = `https://huggingface.co/api/datasets/${EEE_DATASET_REPO}/tree/main`;
const REQUEST_TIMEOUT_MS = 15000;
const REQUEST_MAX_RETRIES = 3;
const REQUEST_RETRY_BASE_DELAY_MS = 750;
const MAX_HTTP_REDIRECTS = 5;

async function main() {
  const startedAt = new Date().toISOString();
  const dryRun = process.argv.includes('--dry-run');
  const loadInstanceInDryRun = process.env.LOAD_INSTANCE_IN_DRY_RUN === '1';
  const configBatchSize = parsePositiveInt(process.env.CONFIG_BATCH_SIZE, DEFAULT_CONFIG_BATCH_SIZE);
  const activeConfigs = getActiveConfigs();

  await ensureCleanOutputDir();

  const metadata = await loadBenchmarkMetadata();
  logInfo('metadata.loaded', { benchmark_card_count: metadata.cards.length, metadata_key_count: metadata.lookup.size });
  const { evaluations, skippedConfigs } = await loadAllEvaluations({ batchSize: configBatchSize, configs: activeConfigs });

  // Load instance-level data in production and optionally for dry-run smoke tests.
  if (!dryRun || loadInstanceInDryRun) {
    await loadInstanceLevelData(evaluations);
  }

  normalizeEvaluations(evaluations);

  const peerRanks = computePeerRanks(evaluations);
  const modelSummaries = buildModelSummaries(evaluations, metadata.lookup);
  const modelCards = buildModelCards(modelSummaries);
  const evalSummaries = buildEvalSummaries(evaluations, metadata.lookup);
  const evalList = buildEvalList(evalSummaries);
  const developerData = buildDeveloperData(modelCards);
  const benchmarkMetadata = metadata.flatMap;
  const manifest = {
    generated_at: startedAt,
    model_count: modelCards.length,
    eval_count: evalSummaries.length,
    config_version: CONFIG_VERSION,
    skipped_config_count: skippedConfigs.length,
    skipped_configs: skippedConfigs,
    source_config_count: activeConfigs.length,
  };

  logInfo('pipeline.summary', {
    dry_run: dryRun,
    evaluations_loaded: evaluations.length,
    model_count: modelCards.length,
    eval_count: evalSummaries.length,
    skipped_config_count: skippedConfigs.length,
  });

  await writeOutputFiles({
    modelCards,
    evalList,
    peerRanks,
    benchmarkMetadata,
    modelSummaries,
    evalSummaries,
    developerData,
    manifest,
  });

  if (!dryRun) {
    await pushToHuggingFace();
  }

  console.log(JSON.stringify({
    dry_run: dryRun,
    model_count: modelCards.length,
    eval_count: evalSummaries.length,
    skipped_configs: skippedConfigs,
    output_dir: outputDir,
  }, null, 2));
}

async function ensureCleanOutputDir() {
  await fs.rm(outputDir, { recursive: true, force: true });
  await fs.mkdir(path.join(outputDir, 'models'), { recursive: true });
  await fs.mkdir(path.join(outputDir, 'evals'), { recursive: true });
  await fs.mkdir(path.join(outputDir, 'developers'), { recursive: true });
}

async function loadBenchmarkMetadata() {
  const entries = await fs.readdir(metadataDir, { withFileTypes: true });
  const files = entries
    .filter((entry) => entry.isFile() && /^benchmark_card_.*\.json$/i.test(entry.name))
    .map((entry) => entry.name)
    .sort();

  const cards = [];
  const lookup = new Map();
  const flatMap = {};

  for (const fileName of files) {
    const fullPath = path.join(metadataDir, fileName);
    const raw = await fs.readFile(fullPath, 'utf8');
    const parsed = JSON.parse(raw);
    const card = parsed?.benchmark_card;
    if (!card) {
      continue;
    }

    const baseName = fileName.replace(/^benchmark_card_/i, '').replace(/\.json$/i, '');
    const keys = candidateBenchmarkKeys(baseName, card?.benchmark_details?.name);
    cards.push({ fileName, baseName, card, keys });

    for (const key of keys) {
      lookup.set(key, card);
      flatMap[key] = card;
    }
  }

  return { cards, lookup, flatMap };
}

async function loadAllEvaluations({ batchSize, configs }) {
  const skippedConfigs = [];
  const evaluations = [];

  for (let index = 0; index < configs.length; index += batchSize) {
    const batch = configs.slice(index, index + batchSize);

    logInfo('config.batch.start', {
      batch_index: Math.floor(index / batchSize),
      batch_size: batch.length,
      configs: batch,
    });

    const results = await Promise.all(batch.map(async (config) => {
      try {
        const startedAt = Date.now();
        const { records, discoveredFiles, discoveryPages } = await loadConfigRecords(config);
        logInfo('config.load.ok', {
          config,
          discovered_data_json_files: discoveredFiles.length,
          discovery_pages: discoveryPages,
          row_count: records.length,
          duration_ms: Date.now() - startedAt,
        });
        return { config, records };
      } catch (error) {
        logInfo('config.load.error', {
          config,
          message: String(error?.message || error),
        });
        return { config, error };
      }
    }));

    for (const result of results) {
      if (result.error) {
        skippedConfigs.push(result.config);
        console.warn(`Skipping config ${result.config}: ${result.error.message}`);
        continue;
      }

      evaluations.push(...result.records.map(mapJsonToEvaluation));
    }

    logInfo('config.batch.done', {
      batch_index: Math.floor(index / batchSize),
      cumulative_evaluations: evaluations.length,
      cumulative_skipped: skippedConfigs.length,
    });
  }

  return { evaluations, skippedConfigs };
}

function getActiveConfigs() {
  const explicitConfigs = parseExplicitConfigs(process.env.CONFIG_NAMES || process.env.CONFIGS);
  if (explicitConfigs.length) {
    return explicitConfigs;
  }

  const limit = parsePositiveInt(process.env.CONFIG_LIMIT, EEE_CONFIGS.length);
  return EEE_CONFIGS.slice(0, Math.max(1, Math.min(limit, EEE_CONFIGS.length)));
}

function parseExplicitConfigs(value) {
  const raw = asString(value).trim();
  if (!raw) {
    return [];
  }

  return raw
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);
}

async function loadConfigRecords(config) {
  let discoveredFiles = [];
  let discoveryPages = 0;
  let discoveryError = null;

  try {
    const discovery = await listDataJsonFilesForConfig(config);
    discoveredFiles = discovery.files;
    discoveryPages = discovery.pages;
  } catch (error) {
    discoveryError = String(error?.message || error);
  }

  logInfo('config.discovery', {
    config,
    data_json_files_found: discoveredFiles.length,
    discovery_pages: discoveryPages,
    discovery_error: discoveryError,
  });

  if (!discoveredFiles.length) {
    throw new Error(`No JSON files discovered under data/${config}`);
  }

  const fetchFile = async (filePath) => {
    const url = `${EEE_DATASET_RAW_BASE}/${filePath}`;
    const { text } = await fetchText(url);
    const parsed = JSON.parse(text);
    parsed.__source_record_url = url;
    return parsed;
  };

  const records = [];
  const FILE_FETCH_CONCURRENCY = 20;
  for (let i = 0; i < discoveredFiles.length; i += FILE_FETCH_CONCURRENCY) {
    const batch = discoveredFiles.slice(i, i + FILE_FETCH_CONCURRENCY);
    const batchRecords = await Promise.all(batch.map(fetchFile));
    records.push(...batchRecords);
  }

  return { records, discoveredFiles, discoveryPages };
}

async function listDataJsonFilesForConfig(config) {
  const apiPath = `data/${config}`;
  let nextUrl = `${EEE_DATASET_TREE_API_BASE}/${apiPath}?recursive=true&expand=true`;
  const files = [];
  let pages = 0;

  while (nextUrl) {
    const { body, headers } = await fetchJsonWithRetry(nextUrl);
    const entries = Array.isArray(body) ? body : [];
    pages += 1;

    for (const entry of entries) {
      const entryPath = asString(entry?.path);
      if (entryPath.endsWith('.json') && !entryPath.endsWith('.jsonl')) {
        files.push(entryPath);
      }
    }

    nextUrl = parseNextLink(headers.link || headers.Link || '');
  }

  files.sort((left, right) => left.localeCompare(right));
  return { files, pages };
}

async function fetchJsonWithRetry(url) {
  let lastError = null;

  for (let attempt = 1; attempt <= REQUEST_MAX_RETRIES; attempt += 1) {
    try {
      const { text, headers } = await fetchText(url);
      return {
        body: JSON.parse(text),
        headers,
      };
    } catch (error) {
      lastError = error;
      if (attempt < REQUEST_MAX_RETRIES) {
        await delay(REQUEST_RETRY_BASE_DELAY_MS * attempt);
      }
    }
  }

  throw lastError;
}

async function fetchText(url, redirectCount = 0) {
  return new Promise((resolve, reject) => {
    const request = https.get(url, {
      timeout: REQUEST_TIMEOUT_MS,
      headers: {
        'user-agent': 'eval-cards-backend-pipeline/1.0',
        accept: '*/*',
      },
    }, (response) => {
      const statusCode = response.statusCode ?? 0;
      const location = asString(response.headers?.location);

      if (statusCode >= 300 && statusCode < 400 && location) {
        if (redirectCount >= MAX_HTTP_REDIRECTS) {
          reject(new Error(`Too many redirects for ${url}`));
          return;
        }

        const nextUrl = new URL(location, url).toString();
        response.resume();
        fetchText(nextUrl, redirectCount + 1).then(resolve).catch(reject);
        return;
      }

      let body = '';
      response.on('data', (chunk) => {
        body += chunk;
      });

      response.on('end', () => {
        if (statusCode >= 200 && statusCode < 300) {
          resolve({ text: body, headers: response.headers || {} });
          return;
        }

        reject(new Error(`HTTP ${statusCode} for ${url}`));
      });
    });

    request.on('timeout', () => {
      request.destroy(new Error(`Timeout fetching ${url}`));
    });

    request.on('error', (error) => {
      reject(error);
    });
  });
}

function parseNextLink(linkHeader) {
  const header = asString(linkHeader);
  if (!header) {
    return null;
  }

  const parts = header.split(',').map((part) => part.trim());
  for (const part of parts) {
    if (!part.includes('rel="next"')) {
      continue;
    }
    const match = part.match(/<([^>]+)>/);
    if (match?.[1]) {
      return match[1];
    }
  }

  return null;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function logInfo(event, details = {}) {
  const payload = {
    ts: new Date().toISOString(),
    event,
    ...details,
  };
  console.log(`[pipeline] ${JSON.stringify(payload)}`);
}

function mapJsonToEvaluation(record) {
  const evaluationResults = Array.isArray(record.evaluation_results) ? record.evaluation_results : [];
  const sourceRecordUrl = asString(record.__source_record_url) || null;
  const passthroughTopLevelFields = extractPassthroughTopLevelFields(record);
  const benchmark = String(record.evaluation_id || '').split('/')[0] || null;
  const firstResult = evaluationResults[0] ?? null;

  return {
    schema_version: asString(record.schema_version),
    evaluation_id: asString(record.evaluation_id),
    retrieved_timestamp: asString(record.retrieved_timestamp),
    benchmark,
    source_data: firstResult?.source_data ?? null,
    source_metadata: record.source_metadata ?? null,
    eval_library: record.eval_library ?? null,
    model_info: record.model_info ?? null,
    generation_config: firstResult?.generation_config ?? null,
    source_record_url: sourceRecordUrl,
    detailed_evaluation_results_meta: normalizeDetailedEvaluationResultsObject(record.detailed_evaluation_results),
    detailed_evaluation_results: resolveDetailedEvaluationResultsUrl({ ...record, source_record_url: sourceRecordUrl }),
    passthrough_top_level_fields: passthroughTopLevelFields,
    evaluation_results: normalizeEvaluationResults(evaluationResults),
  };
}

function extractPassthroughTopLevelFields(rawRecord) {
  if (!rawRecord || typeof rawRecord !== 'object' || Array.isArray(rawRecord)) {
    return null;
  }

  const knownKeys = new Set([
    'schema_version',
    'evaluation_id',
    'retrieved_timestamp',
    'source_metadata',
    'eval_library',
    'model_info',
    'evaluation_results',
    'detailed_evaluation_results',
    '__source_record_url',
  ]);

  const entries = Object.entries(rawRecord).filter(([key]) => !knownKeys.has(key));
  return entries.length ? Object.fromEntries(entries) : null;
}

function resolveDetailedEvaluationResultsUrl(row) {
  const sourceRecordUrl = asString(row?.source_record_url);
  const explicit = normalizeDetailedEvaluationResultsValue(row?.detailed_evaluation_results, sourceRecordUrl);
  if (explicit) {
    if (/^https?:\/\//i.test(explicit)) {
      return explicit;
    }
    const cleaned = explicit.replace(/^\/+/, '');
    return `${EEE_DATASET_RAW_BASE}/${cleaned}`;
  }

  if (!sourceRecordUrl || !sourceRecordUrl.endsWith('.json')) {
    return null;
  }

  // EEE commonly stores companion files as *_samples.jsonl.
  return `${sourceRecordUrl.slice(0, -5)}_samples.jsonl`;
}

function normalizeDetailedEvaluationResultsValue(value, sourceRecordUrl) {
  if (!value) {
    return '';
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (trimmed.startsWith('{') && trimmed.includes('file_path')) {
      const filePathMatch = trimmed.match(/file_path'?:\s*'([^']+)'/i) || trimmed.match(/"file_path"\s*:\s*"([^"]+)"/i);
      if (filePathMatch?.[1]) {
        const filePath = filePathMatch[1];
        if (/^https?:\/\//i.test(filePath)) {
          return filePath;
        }
        if (filePath.startsWith('data/')) {
          return filePath;
        }

        const sourceUrl = asString(sourceRecordUrl);
        if (sourceUrl && /^https?:\/\//i.test(sourceUrl)) {
          const baseDir = sourceUrl.slice(0, sourceUrl.lastIndexOf('/') + 1);
          return `${baseDir}${filePath.replace(/^\/+/, '')}`;
        }

        return filePath;
      }
    }
    return value;
  }

  if (typeof value === 'object') {
    const filePath = asString(value.file_path || value.path || value.url || '');
    if (!filePath) {
      return '';
    }
    if (/^https?:\/\//i.test(filePath)) {
      return filePath;
    }
    if (filePath.startsWith('data/')) {
      return filePath;
    }

    const sourceUrl = asString(sourceRecordUrl);
    if (sourceUrl && /^https?:\/\//i.test(sourceUrl)) {
      const baseDir = sourceUrl.slice(0, sourceUrl.lastIndexOf('/') + 1);
      return `${baseDir}${filePath.replace(/^\/+/, '')}`;
    }

    return filePath;
  }

  return '';
}

function normalizeDetailedEvaluationResultsObject(value) {
  if (!value) {
    return null;
  }

  if (typeof value === 'object') {
    const normalized = toJsonSafeValue(value);
    if (normalized && typeof normalized === 'object' && normalized.entries && typeof normalized.entries === 'object') {
      return normalized.entries;
    }
    return normalized;
  }

  if (typeof value === 'string') {
    const parsed = safeJsonParse(value, null);
    if (parsed && typeof parsed === 'object') {
      return parsed;
    }

    const filePathMatch = value.match(/file_path'?:\s*'([^']+)'/i) || value.match(/"file_path"\s*:\s*"([^"]+)"/i);
    const formatMatch = value.match(/format'?:\s*'([^']+)'/i) || value.match(/"format"\s*:\s*"([^"]+)"/i);
    const rowsMatch = value.match(/total_rows'?:\s*([0-9]+)/i) || value.match(/"total_rows"\s*:\s*([0-9]+)/i);

    if (filePathMatch?.[1] || formatMatch?.[1] || rowsMatch?.[1]) {
      return {
        file_path: filePathMatch?.[1] ?? null,
        format: formatMatch?.[1] ?? null,
        total_rows: rowsMatch?.[1] ? Number(rowsMatch[1]) : null,
      };
    }
  }

  return null;
}

function toJsonSafeValue(value) {
  if (typeof value === 'bigint') {
    const asNumber = Number(value);
    return Number.isSafeInteger(asNumber) ? asNumber : String(value);
  }

  if (Array.isArray(value)) {
    return value.map((entry) => toJsonSafeValue(entry));
  }

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, toJsonSafeValue(entry)])
    );
  }

  return value;
}

function normalizeEvaluationResults(results) {
  return (results ?? []).map((result) => {
    const normalized = { ...result };
    if (normalized?.score_details?.details && typeof normalized.score_details.details === 'string') {
      normalized.score_details = {
        ...normalized.score_details,
        details: safeJsonParse(normalized.score_details.details, normalized.score_details.details),
      };
    }
    return normalized;
  });
}

async function loadInstanceLevelData(evaluations) {
  // Load companion JSONL files for records discovered in data/*.
  const batchSize = 10;
  let withInstanceData = 0;
  let missingInstanceData = 0;
  
  for (let i = 0; i < evaluations.length; i += batchSize) {
    const batch = evaluations.slice(i, i + batchSize);
    
    const promises = batch.map(async (evaluation) => {
      try {
        const instanceData = await fetchInstanceData(evaluation);
        if (instanceData) {
          evaluation.instance_level_data = instanceData;
          withInstanceData += 1;
        } else {
          missingInstanceData += 1;
        }
      } catch {
        missingInstanceData += 1;
      }
    });
    
    await Promise.all(promises);

    logInfo('instance.batch.progress', {
      processed: Math.min(i + batch.length, evaluations.length),
      total: evaluations.length,
      with_instance_data: withInstanceData,
      missing_instance_data: missingInstanceData,
    });
  }

  logInfo('instance.load.summary', {
    total: evaluations.length,
    with_instance_data: withInstanceData,
    missing_instance_data: missingInstanceData,
  });
}

async function fetchInstanceData(evaluation) {
  const jsonlCandidates = candidateInstanceUrls(evaluation);
  if (!jsonlCandidates.length) {
    return null;
  }

  for (const jsonlUrl of jsonlCandidates) {
    try {
      const data = await fetchJsonlFile(jsonlUrl);
      if (data && Array.isArray(data) && data.length > 0) {
        return {
          interaction_type: inferInteractionType(data),
          instance_count: data.length,
          source_url: jsonlUrl,
          instance_examples: pickRandomExamples(data, 5),
        };
      }
    } catch {
      continue;
    }
  }
  
  return null;
}

function candidateInstanceUrls(evaluation) {
  const candidates = new Set();
  const explicit = asString(evaluation?.detailed_evaluation_results);
  const sourceRecordUrl = asString(evaluation?.source_record_url);

  if (explicit) {
    candidates.add(explicit);
  }

  if (sourceRecordUrl && sourceRecordUrl.endsWith('.json')) {
    const base = sourceRecordUrl.slice(0, -5);
    candidates.add(`${base}_samples.jsonl`);
    candidates.add(`${base}.jsonl`);
  }

  return [...candidates];
}

async function fetchJsonlFile(url) {
  const { text } = await fetchText(url);
  const lines = text.trim().split('\n').filter(Boolean);
  return lines.map((line) => safeJsonParse(line, null)).filter(Boolean);
}

function inferInteractionType(instances) {
  if (!Array.isArray(instances) || instances.length === 0) {
    return 'unknown';
  }
  
  const first = instances[0];
  if (first.interactions || first.messages) {
    return 'multi_turn';
  } else if (first.input && first.output && first.evaluation) {
    return 'single_turn';
  } else if (first.tool_calls || first.tool_use) {
    return 'agentic';
  }
  return 'unknown';
}

function pickRandomExamples(items, count) {
  if (!Array.isArray(items) || !items.length || count <= 0) {
    return [];
  }

  if (items.length <= count) {
    return items;
  }

  const copy = [...items];
  // Fisher-Yates shuffle then slice for unbiased random examples.
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }

  return copy.slice(0, count);
}

function normalizeEvaluations(evaluations) {
  for (const evaluation of evaluations) {
    const identity = getCanonicalModelIdentity(evaluation.model_info);
    evaluation.model_info = {
      ...evaluation.model_info,
      normalized_id: identity.normalizedId,
      family_id: identity.familyId,
      family_slug: identity.familySlug,
      family_name: identity.familyName,
      variant_key: identity.variantKey,
      variant_label: identity.variantLabel,
      model_route_id: identity.modelRouteId,
    };
  }
}

function normalizeModelInfo(modelInfo) {
  const rawId = asString(modelInfo?.id || modelInfo?.name || 'unknown/unknown');
  const fallbackDeveloper = asString(modelInfo?.developer || rawId.split('/')[0] || 'unknown');
  const parts = rawId.includes('/') ? rawId.split('/') : [slugifyDeveloper(fallbackDeveloper), rawId];
  const rawDeveloper = parts.length > 1 ? parts[0] : slugifyDeveloper(fallbackDeveloper);
  const rawModelName = parts.length > 1 ? parts.slice(1).join('/') : parts[0];
  const match = rawModelName.match(VERSION_SUFFIX_REGEX);
  const baseSlug = match ? match[1] : rawModelName;
  const versionDate = match ? match[2] : null;
  const qualifier = match ? match[3] ?? null : null;

  return {
    rawId,
    developer: asString(modelInfo?.developer || humanizeSlug(rawDeveloper)),
    developerSlug: slugifyDeveloper(rawDeveloper),
    modelName: asString(modelInfo?.name || humanizeSlug(baseSlug)),
    rawModelName,
    familySlug: slugifyModelSegment(baseSlug),
    versionDate,
    qualifier,
  };
}

function getCanonicalModelIdentity(modelInfo) {
  const normalized = normalizeModelInfo(modelInfo);
  const familyId = `${normalized.developerSlug}/${normalized.familySlug}`;
  const normalizedId = normalized.rawId.includes('/')
    ? `${normalized.developerSlug}/${normalized.rawModelName}`
    : `${normalized.developerSlug}/${normalized.rawId}`;
  const variantParts = [normalized.versionDate, normalized.qualifier].filter(Boolean);
  const variantKey = variantParts.length ? variantParts.join('-') : 'default';
  const variantLabel = variantParts.length ? variantParts.join(' ') : 'Default';

  return {
    normalizedId,
    familyId,
    familySlug: normalized.familySlug,
    familyName: normalized.modelName,
    modelRouteId: familyId.replace(/\//g, '__'),
    variantKey,
    variantLabel,
  };
}

function groupByModelFamily(evaluations) {
  const grouped = new Map();
  for (const evaluation of evaluations) {
    const key = evaluation.model_info?.family_id || getCanonicalModelIdentity(evaluation.model_info).familyId;
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key).push(evaluation);
  }
  return grouped;
}

function groupByBenchmark(evaluations) {
  const grouped = new Map();
  for (const evaluation of evaluations) {
    for (const result of evaluation.evaluation_results ?? []) {
      const evalSummaryId = getEvalSummaryId(evaluation, result);
      const lowerIsBetter = Boolean(result?.metric_config?.lower_is_better);
      const score = numericScore(result?.score_details?.score);
      if (!Number.isFinite(score)) {
        continue;
      }
      if (!grouped.has(evalSummaryId)) {
        grouped.set(evalSummaryId, {
          eval_summary_id: evalSummaryId,
          benchmark: evaluation.benchmark || result?.source_data?.dataset_name || result?.evaluation_name || 'unknown',
          evaluation_name: result?.evaluation_name || 'unknown',
          lower_is_better: lowerIsBetter,
          metric_config: result?.metric_config ?? null,
          source_data: result?.source_data ?? null,
          model_results: [],
        });
      }

      grouped.get(evalSummaryId).model_results.push({
        model_id: evaluation.model_info?.family_id,
        model_route_id: evaluation.model_info?.model_route_id,
        model_name: evaluation.model_info?.family_name || evaluation.model_info?.name,
        developer: evaluation.model_info?.developer,
        raw_model_id: evaluation.model_info?.id,
        score,
        evaluation_id: evaluation.evaluation_id,
        retrieved_timestamp: evaluation.retrieved_timestamp,
        source_record_url: evaluation.source_record_url ?? null,
        detailed_evaluation_results: evaluation.detailed_evaluation_results ?? null,
        detailed_evaluation_results_meta: evaluation.detailed_evaluation_results_meta ?? null,
        passthrough_top_level_fields: evaluation.passthrough_top_level_fields ?? null,
        instance_level_data: evaluation.instance_level_data ?? null,
      });
    }
  }
  return grouped;
}

function computePeerRanks(evaluations) {
  const grouped = groupByBenchmark(evaluations);
  const peerRanks = {};

  for (const [evalSummaryId, summary] of grouped.entries()) {
    const sorted = [...summary.model_results].sort((left, right) => {
      return summary.lower_is_better ? left.score - right.score : right.score - left.score;
    });

    let previousScore = null;
    let currentRank = 0;

    peerRanks[evalSummaryId] = {};

    for (let index = 0; index < sorted.length; index += 1) {
      const entry = sorted[index];
      if (previousScore === null || entry.score !== previousScore) {
        currentRank += 1;
        previousScore = entry.score;
      }
      peerRanks[evalSummaryId][entry.model_id] = {
        position: currentRank,
        total: sorted.length,
      };
    }
  }

  return peerRanks;
}

function buildModelSummaries(evaluations, metadataLookup) {
  const grouped = groupByModelFamily(evaluations);
  const summaries = [];

  for (const [familyId, familyEvaluations] of grouped.entries()) {
    summaries.push(createModelFamilySummary(familyId, familyEvaluations, metadataLookup));
  }

  return summaries.sort((left, right) => left.model_family_name.localeCompare(right.model_family_name));
}

function createModelFamilySummary(familyId, evaluations, metadataLookup) {
  const first = evaluations[0];
  const categories = {};
  const variantMap = new Map();
  const rawModelIds = new Set();
  const timestamps = [];

  for (const evaluation of evaluations) {
    rawModelIds.add(evaluation.model_info?.id);
    const category = inferCategoryFromBenchmark(evaluation.benchmark);
    const attached = attachBenchmarkCard(evaluation, metadataLookup);
    if (!categories[category]) {
      categories[category] = [];
    }
    categories[category].push(attached);

    const variantKey = evaluation.model_info?.variant_key || 'default';
    if (!variantMap.has(variantKey)) {
      variantMap.set(variantKey, {
        variant_key: variantKey,
        variant_label: evaluation.model_info?.variant_label || 'Default',
        raw_model_ids: new Set(),
        evaluation_count: 0,
        last_updated: null,
      });
    }
    const variant = variantMap.get(variantKey);
    variant.raw_model_ids.add(evaluation.model_info?.id);
    variant.evaluation_count += 1;
    variant.last_updated = maxIsoLikeTimestamp(variant.last_updated, epochStringToIso(evaluation.retrieved_timestamp));

    timestamps.push(epochStringToIso(evaluation.retrieved_timestamp));
  }

  const categoriesCovered = Object.keys(categories).sort();
  const modelInfo = {
    ...first.model_info,
    name: first.model_info?.family_name || first.model_info?.name,
  };

  return {
    model_info: modelInfo,
    model_family_id: familyId,
    model_route_id: first.model_info?.model_route_id,
    model_family_name: first.model_info?.family_name || first.model_info?.name || familyId,
    raw_model_ids: [...rawModelIds].filter(Boolean).sort(),
    evaluations_by_category: categories,
    total_evaluations: evaluations.length,
    last_updated: timestamps.filter(Boolean).sort().at(-1) ?? null,
    categories_covered: categoriesCovered,
    variants: [...variantMap.values()]
      .map((variant) => ({
        ...variant,
        raw_model_ids: [...variant.raw_model_ids].filter(Boolean).sort(),
      }))
      .sort((left, right) => left.variant_key.localeCompare(right.variant_key)),
  };
}

function attachBenchmarkCard(evaluation, metadataLookup) {
  const keys = candidateBenchmarkKeys(evaluation.benchmark, evaluation.source_data?.dataset_name);
  let benchmarkCard = null;
  for (const key of keys) {
    if (metadataLookup.has(key)) {
      benchmarkCard = metadataLookup.get(key);
      break;
    }
  }

  return {
    ...evaluation,
    benchmark_card: benchmarkCard,
  };
}

function createEvaluationCard(summary) {
  const benchmarkCount = countDistinctBenchmarks(summary);
  const scoreStats = aggregateSummaryScores(summary);

  return {
    model_family_id: summary.model_family_id,
    model_route_id: summary.model_route_id,
    model_family_name: summary.model_family_name,
    developer: summary.model_info?.developer || 'Unknown',
    total_evaluations: summary.total_evaluations,
    benchmark_count: benchmarkCount,
    categories_covered: summary.categories_covered,
    last_updated: summary.last_updated,
    variants: summary.variants.map((variant) => ({
      variant_key: variant.variant_key,
      variant_label: variant.variant_label,
      evaluation_count: variant.evaluation_count,
      raw_model_ids: variant.raw_model_ids,
      last_updated: variant.last_updated,
    })),
    score_summary: scoreStats,
  };
}

function buildModelCards(modelSummaries) {
  return modelSummaries
    .map((summary) => createEvaluationCard(summary))
    .sort((left, right) => {
      if (right.benchmark_count !== left.benchmark_count) {
        return right.benchmark_count - left.benchmark_count;
      }
      return left.model_family_name.localeCompare(right.model_family_name);
    });
}

function buildEvalSummaries(evaluations, metadataLookup) {
  const grouped = groupByBenchmark(evaluations);
  const summaries = [];

  for (const summary of grouped.values()) {
    const metadata = lookupBenchmarkCard(metadataLookup, summary.benchmark, summary.source_data?.dataset_name, summary.evaluation_name);
    const modelResults = summary.model_results.sort((left, right) => {
      return summary.lower_is_better ? left.score - right.score : right.score - left.score;
    });

    summaries.push({
      eval_summary_id: summary.eval_summary_id,
      benchmark: summary.benchmark,
      evaluation_name: summary.evaluation_name,
      lower_is_better: summary.lower_is_better,
      metric_config: summary.metric_config,
      source_data: summary.source_data,
      benchmark_card: metadata,
      models_count: modelResults.length,
      model_results: modelResults,
    });
  }

  return summaries.sort((left, right) => {
    if (right.models_count !== left.models_count) {
      return right.models_count - left.models_count;
    }
    return left.eval_summary_id.localeCompare(right.eval_summary_id);
  });
}

function buildEvalList(evalSummaries) {
  const evals = evalSummaries.map((summary) => toBenchmarkEvalListItem(summary));
  const modelIds = new Set();
  for (const summary of evalSummaries) {
    for (const modelResult of summary.model_results) {
      modelIds.add(modelResult.model_id);
    }
  }

  return {
    evals,
    totalModels: modelIds.size,
  };
}

function toBenchmarkEvalListItem(summary) {
  return {
    eval_summary_id: summary.eval_summary_id,
    benchmark: summary.benchmark,
    evaluation_name: summary.evaluation_name,
    lower_is_better: summary.lower_is_better,
    models_count: summary.models_count,
    benchmark_card: summary.benchmark_card,
    source_data: summary.source_data,
    metric_config: summary.metric_config,
    top_score: summary.model_results[0]?.score ?? null,
  };
}

function buildDeveloperData(modelCards) {
  const grouped = new Map();
  for (const card of modelCards) {
    const developer = card.developer || 'Unknown';
    if (!grouped.has(developer)) {
      grouped.set(developer, []);
    }
    grouped.get(developer).push(card);
  }

  const developers = [...grouped.entries()]
    .map(([developer, models]) => ({ developer, model_count: models.length }))
    .sort((left, right) => {
      if (right.model_count !== left.model_count) {
        return right.model_count - left.model_count;
      }
      return left.developer.localeCompare(right.developer);
    });

  const summaries = developers.map(({ developer }) => ({
    developer,
    slug: slugifyDeveloperSummary(developer),
    models: grouped.get(developer).sort((left, right) => left.model_family_name.localeCompare(right.model_family_name)),
  }));

  return { developers, summaries };
}

async function writeOutputFiles({ modelCards, evalList, peerRanks, benchmarkMetadata, modelSummaries, evalSummaries, developerData, manifest }) {
  await writeJson(path.join(outputDir, 'model-cards.json'), modelCards);
  await writeJson(path.join(outputDir, 'eval-list.json'), evalList);
  await writeJson(path.join(outputDir, 'peer-ranks.json'), peerRanks);
  await writeJson(path.join(outputDir, 'benchmark-metadata.json'), benchmarkMetadata);
  await writeJson(path.join(outputDir, 'developers.json'), developerData.developers);
  await writeJson(path.join(outputDir, 'manifest.json'), manifest);

  for (const summary of modelSummaries) {
    await writeJson(path.join(outputDir, 'models', `${summary.model_route_id}.json`), summary);
  }

  for (const summary of evalSummaries) {
    await writeJson(path.join(outputDir, 'evals', `${summary.eval_summary_id}.json`), summary);
  }

  for (const summary of developerData.summaries) {
    await writeJson(path.join(outputDir, 'developers', `${summary.slug}.json`), {
      developer: summary.developer,
      models: summary.models,
    });
  }
}

async function pushToHuggingFace() {
  const accessToken = process.env.HF_TOKEN;
  if (!accessToken) {
    throw new Error('HF_TOKEN is required unless --dry-run is used');
  }

  try {
    await createRepo({
      repo: DATASET_REPO,
      accessToken,
      private: false,
    });
  } catch (error) {
    if (!String(error?.message || '').includes('already exists')) {
      console.warn(`createRepo warning: ${error.message}`);
    }
  }

  await uploadFiles({
    repo: DATASET_REPO,
    accessToken,
    files: [pathToFileURL(outputDir)],
    commitTitle: `Pipeline sync ${new Date().toISOString()}`,
  });
}

function lookupBenchmarkCard(metadataLookup, ...values) {
  const keys = candidateBenchmarkKeys(...values);
  for (const key of keys) {
    if (metadataLookup.has(key)) {
      return metadataLookup.get(key);
    }
  }
  return null;
}

function candidateBenchmarkKeys(...values) {
  const keys = new Set();
  for (const value of values) {
    const stringValue = asString(value);
    if (!stringValue) {
      continue;
    }
    keys.add(normalizeBenchmarkKey(stringValue));
    keys.add(normalizeBenchmarkKey(stringValue.replace(/^benchmark_card_/i, '')));
    keys.add(normalizeBenchmarkKey(stringValue.replace(/[_-]+/g, ' ')));
  }
  return [...keys].filter(Boolean);
}

function normalizeBenchmarkKey(value) {
  return asString(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function slugify(value) {
  return normalizeBenchmarkKey(value);
}

function getEvalSummaryId(evaluation, result) {
  const benchmarkKey = evaluation.benchmark || result?.source_data?.dataset_name || result?.evaluation_name;
  return slugify(`${benchmarkKey}__${result?.evaluation_name || 'unknown'}`);
}

function inferCategoryFromBenchmark(benchmarkName) {
  const key = normalizeBenchmarkKey(benchmarkName);

  if (!key) {
    return 'other';
  }
  if (/(math|gsm|gpqa|mmlu|medqa|legalbench|boolq|hellaswag|quac|cnn_dailymail|civilcomments|ifeval|musr)/.test(key)) {
    return 'reasoning';
  }
  if (/(appworld|swe_bench|tau_bench|browsecomp|agent|livecodebench)/.test(key)) {
    return 'agentic';
  }
  if (/(reward_bench|hfopenllm|helm)/.test(key)) {
    return 'general';
  }
  return 'other';
}

function countDistinctBenchmarks(summary) {
  const benchmarks = new Set();
  for (const evaluations of Object.values(summary.evaluations_by_category)) {
    for (const evaluation of evaluations) {
      if (evaluation.benchmark) {
        benchmarks.add(evaluation.benchmark);
      }
    }
  }
  return benchmarks.size;
}

function aggregateSummaryScores(summary) {
  const scores = [];
  for (const evaluations of Object.values(summary.evaluations_by_category)) {
    for (const evaluation of evaluations) {
      for (const result of evaluation.evaluation_results ?? []) {
        const score = numericScore(result?.score_details?.score);
        if (Number.isFinite(score)) {
          scores.push(score);
        }
      }
    }
  }

  if (!scores.length) {
    return { count: 0, min: null, max: null, average: null };
  }

  const total = scores.reduce((sum, value) => sum + value, 0);
  return {
    count: scores.length,
    min: Math.min(...scores),
    max: Math.max(...scores),
    average: total / scores.length,
  };
}

function numericScore(value) {
  if (typeof value === 'number') {
    return value;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : NaN;
}

function asString(value) {
  if (value === null || value === undefined) {
    return '';
  }
  return String(value);
}

function safeJsonParse(value, fallback) {
  if (value === null || value === undefined || value === '') {
    return fallback;
  }
  if (typeof value !== 'string') {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function humanizeSlug(value) {
  return asString(value)
    .split(/[-_\/]+/)
    .filter(Boolean)
    .map((part) => part.length <= 3 && /\d/.test(part) ? part.toUpperCase() : capitalize(part))
    .join(' ');
}

function capitalize(value) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

function slugifyDeveloper(value) {
  return asString(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'unknown';
}

function slugifyModelSegment(value) {
  return asString(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'unknown';
}

function slugifyDeveloperSummary(value) {
  return slugifyDeveloper(value);
}

function epochStringToIso(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  return new Date(numeric * 1000).toISOString();
}

function maxIsoLikeTimestamp(left, right) {
  if (!left) {
    return right;
  }
  if (!right) {
    return left;
  }
  return left > right ? left : right;
}

async function writeJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
