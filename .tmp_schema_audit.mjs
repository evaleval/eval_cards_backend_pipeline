import { DuckDBConnection } from '@duckdb/node-api';

const conn = await DuckDBConnection.create();
await conn.run('INSTALL httpfs; LOAD httpfs;');

const urls = [
  'https://huggingface.co/datasets/evaleval/EEE_datastore/raw/main/data/fibble1_arena/anthropic/claude-haiku-4-5-20251001/fdf971b5-a991-48e6-931e-61961ef3fd7c.json',
  'https://huggingface.co/datasets/evaleval/EEE_datastore/raw/main/data/ace/openai/gpt-5/b453856c-f4ce-4881-8dd5-5d22b1a6d201.json'
];

const sql = `SELECT * FROM read_json_auto([${urls.map((u) => `'${u}'`).join(', ')}], filename = true)`;
const reader = await conn.runAndReadAll(sql);
console.log(JSON.stringify(reader.columnNames(), null, 2));
conn.closeSync();
