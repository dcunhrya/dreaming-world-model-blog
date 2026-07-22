import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runOfflineExperiments } from '../src/dreaming/runExperiment';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outPath = join(__dirname, '../public/data/dyna_results.json');

console.log('Running offline Dyna-Q experiments...');
const results = runOfflineExperiments();

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(results, null, 2));
console.log(`Wrote ${outPath}`);
