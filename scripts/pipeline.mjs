import { DuckDBConnection } from '@duckdb/node-api';
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
const EEE_DATASET_RESOLVE_BASE = `https://huggingface.co/datasets/${EEE_DATASET_REPO}/resolve/main`;
const EEE_DATASET_TREE_API_BASE = `https://huggingface.co/api/datasets/${EEE_DATASET_REPO}/tree/main`;
const REQUEST_TIMEOUT_MS = 15000;
const REQUEST_MAX_RETRIES = 3;
const REQUEST_RETRY_BASE_DELAY_MS = 750;
const JSON_LOAD_CHUNK_SIZE = 250;

async function main() {
  const startedAt = new Date().toISOString();
  const dryRun = process.argv.includes('--dry-run');
  const configBatchSize = parsePositiveInt(process.env.CONFIG_BATCH_SIZE, DEFAULT_CONFIG_BATCH_SIZE);
  const activeConfigs = getActiveConfigs();

  await ensureCleanOutputDir();

  const metadata = await loadBenchmarkMetadata();
  logInfo('metadata.loaded', { benchmark_card_count: metadata.cards.length, metadata_key_count: metadata.lookup.size });
  const { evaluations, skippedConfigs } = await loadAllEvaluations({ batchSize: configBatchSize, configs: activeConfigs });

  // Load instance-level data only in production (skip in dry-run for speed)
  if (!dryRun) {
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
  const connection = await DuckDBConnection.create();
  const skippedConfigs = [];
  const evaluations = [];

  try {
    await connection.run(`INSTALL httpfs; LOAD httpfs;`);
  } catch (error) {
    console.warn(`DuckDB httpfs extension init warning: ${error.message}`);
  }

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
        const { rows, discoveredFiles, discoveryPages } = await loadConfigRows(connection, config);
        logInfo('config.load.ok', {
          config,
          discovered_data_json_files: discoveredFiles.length,
          discovery_pages: discoveryPages,
          row_count: rows.length,
          duration_ms: Date.now() - startedAt,
        });
        return { config, rows };
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

      evaluations.push(...result.rows.map(mapDuckDbRowToEvaluation));
    }

    logInfo('config.batch.done', {
      batch_index: Math.floor(index / batchSize),
      cumulative_evaluations: evaluations.length,
      cumulative_skipped: skippedConfigs.length,
    });
  }

  connection.closeSync();
  return { evaluations, skippedConfigs };
}

function getActiveConfigs() {
  const limit = parsePositiveInt(process.env.CONFIG_LIMIT, EEE_CONFIGS.length);
  return EEE_CONFIGS.slice(0, Math.max(1, Math.min(limit, EEE_CONFIGS.length)));
}

async function loadConfigRows(connection, config) {
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

  const urls = discoveredFiles.map((entry) => `${EEE_DATASET_RESOLVE_BASE}/${entry}`);
  const rows = [];

  for (let offset = 0; offset < urls.length; offset += JSON_LOAD_CHUNK_SIZE) {
    const chunk = urls.slice(offset, offset + JSON_LOAD_CHUNK_SIZE);
    const sql = `
      SELECT
        schema_version,
        evaluation_id,
        retrieved_timestamp,
        to_json(source_metadata) AS source_metadata_json,
        to_json(eval_library) AS eval_library_json,
        to_json(model_info) AS model_info_json,
        to_json(evaluation_results) AS evaluation_results_json,
        filename AS source_record_url
      FROM read_json_auto(${toSqlStringArray(chunk)}, filename = true)
    `;
    const reader = await connection.runAndReadAll(sql);
    rows.push(...rowsFromReader(reader));
  }

  return {
    rows,
    discoveredFiles,
    discoveryPages,
  };
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

function toSqlStringArray(values) {
  return `[${values.map((value) => `'${asString(value).replace(/'/g, "''")}'`).join(', ')}]`;
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

async function fetchText(url) {
  return new Promise((resolve, reject) => {
    const request = https.get(url, {
      timeout: REQUEST_TIMEOUT_MS,
      headers: {
        'user-agent': 'eval-cards-backend-pipeline/1.0',
        accept: 'application/json',
      },
    }, (response) => {
      const statusCode = response.statusCode ?? 0;
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

function rowsFromReader(reader) {
  const columnNames = reader.columnNames();
  const rows = reader.getRows();
  return rows.map((row) => Object.fromEntries(columnNames.map((name, index) => [name, row[index]])));
}

function mapDuckDbRowToEvaluation(row) {
  const evaluationResults = safeJsonParse(row.evaluation_results_json, []);
  const sourceMetadata = safeJsonParse(row.source_metadata_json, null);
  const evalLibrary = safeJsonParse(row.eval_library_json, null);
  const modelInfo = safeJsonParse(row.model_info_json, null);
  const benchmark = String(row.evaluation_id || '').split('/')[0] || null;
  const firstResult = evaluationResults[0] ?? null;

  return {
    schema_version: asString(row.schema_version),
    evaluation_id: asString(row.evaluation_id),
    retrieved_timestamp: asString(row.retrieved_timestamp),
    benchmark,
    source_data: firstResult?.source_data ?? null,
    source_metadata: sourceMetadata,
    eval_library: evalLibrary,
    model_info: modelInfo,
    generation_config: firstResult?.generation_config ?? null,
    source_record_url: asString(row.source_record_url) || null,
    detailed_evaluation_results: deriveJsonlUrl(asString(row.source_record_url)),
    evaluation_results: normalizeEvaluationResults(evaluationResults),
  };
}

function deriveJsonlUrl(sourceRecordUrl) {
  const url = asString(sourceRecordUrl);
  if (!url || !url.endsWith('.json')) {
    return null;
  }
  return `${url.slice(0, -5)}.jsonl`;
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
  const jsonlUrl = evaluation?.detailed_evaluation_results;
  if (!jsonlUrl) {
    return null;
  }

  try {
    const data = await fetchJsonlFile(jsonlUrl);
    if (data && Array.isArray(data) && data.length > 0) {
      return {
        interaction_type: inferInteractionType(data),
        instance_count: data.length,
        source_url: jsonlUrl,
        instances: data,
      };
    }
  } catch {
    return null;
  }
  
  return null;
}

async function fetchJsonlFile(url) {
  const text = await fetchText(url);
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
