import { DuckDBConnection } from '@duckdb/node-api';

const url = 'https://huggingface.co/api/datasets/evaleval/EEE_datastore/parquet/ace/train/0.parquet';
const connection = await DuckDBConnection.create();
await connection.run('INSTALL httpfs; LOAD httpfs;');

const sql = `SELECT * FROM read_parquet('${url}') LIMIT 1`;
const sample = await connection.runAndReadAll(sql);
const columnNames = sample.columnNames();
const rows = sample.getRows();

console.log('=== Column Names ===');
console.log(columnNames);

if (rows.length > 0) {
  const row = rows[0];
  console.log('\n=== All Fields in EEE Row ===');
  columnNames.forEach((name, i) => {
    const val = row[i];
    let desc = typeof val;
    if (val === null) desc = 'null';
    else if (Array.isArray(val)) desc = `array[${val.length}]`;
    else if (typeof val === 'object') desc = 'object';
    console.log(`  ${name}: ${desc}`);
  });
  
  console.log('\n=== instance_data ===');
  const instanceDataIdx = columnNames.indexOf('instance_data');
  if (instanceDataIdx >= 0 && row[instanceDataIdx]) {
    console.log('PRESENT - keys:', Object.keys(row[instanceDataIdx]));
    const inst = row[instanceDataIdx];
    console.log(JSON.stringify(inst, null, 2).substring(0, 800));
  } else {
    console.log('MISSING');
  }
  
  console.log('\n=== Captured in current pipeline ===');
  columnNames.forEach((name, i) => {
    console.log(`${name}: ${row[i] ? 'YES' : 'NO'}`);
  });
  
  console.log('\n=== evaluation_results[0] shape ===');
  const evalIdx = columnNames.indexOf('evaluation_results');
  if (evalIdx >= 0 && row[evalIdx]?.[0]) {
    const res = row[evalIdx][0];
    console.log('Keys:', Object.keys(res).sort());
    console.log('\nSample:', JSON.stringify(res, null, 2).substring(0, 1200));
  }
}

connection.closeSync();
