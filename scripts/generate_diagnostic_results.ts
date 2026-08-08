import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  diagnosticResultsChecksum,
  runDiagnosticOfflineExperiments,
} from '../src/dreaming/diagnostic/runExperiment';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outPath = join(__dirname, '../public/data/diagnostic_results.json');

const results = runDiagnosticOfflineExperiments();
const checksum = diagnosticResultsChecksum(results);

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify({ ...results, checksum }, null, 2));

console.log(`Wrote ${outPath}`);
console.log(`Checksum: ${checksum}`);
console.log(`Agents: ${results.finalMetricsByAgent.map((a) => a.agent).join(', ')}`);
