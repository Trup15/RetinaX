import React, { useState } from 'react';
import { GENERALIZATION_BENCHMARKS, CONFUSION_MATRICES } from '../data/researchBenchmarks';
import { DR_STAGES, DRStage } from '../types/pipeline';
import { Database, AlertOctagon, TrendingDown, ArrowRight, ShieldAlert, BarChart2, Info } from 'lucide-react';

export const CrossDatasetShiftExplorer: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<'APTOS_2019' | 'IDRiD' | 'DDR'>('APTOS_2019');
  const [showNormalized, setShowNormalized] = useState(false);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; count: number } | null>(null);

  const activeBenchmark =
    GENERALIZATION_BENCHMARKS.find((b) => b.dataset === selectedDataset) ||
    GENERALIZATION_BENCHMARKS[0];

  const currentMatrix = CONFUSION_MATRICES[selectedDataset];

  // Calculate row sums for normalization
  const rowSums = currentMatrix.map((row) => row.reduce((a, b) => a + b, 0));

  return (
    <div className="space-y-6">
      {/* Overview Context Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="text-xs font-mono text-cyan-400 font-semibold uppercase tracking-wider mb-1">
              Objective 5 · Cross-Dataset Robustness & Domain Shift
            </div>
            <h2 className="text-xl font-bold text-slate-100">
              Generalization Evaluation Across Independent Cohorts
            </h2>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
              Addressing <strong className="text-slate-300">Research Gap 3 (RG3)</strong>: Unlike models tested solely on same-source held-out validation sets, the EfficientNet-B3 classifier trained on <strong className="text-cyan-300">APTOS 2019</strong> was frozen and evaluated <strong className="text-amber-300">completely unchanged</strong> on the external Indian cohort (<strong className="text-emerald-300">IDRiD</strong>) and the broad diverse <strong className="text-purple-300">DDR</strong> benchmark to measure real-world clinical transferability.
            </p>
          </div>

          <div className="flex items-center gap-2 bg-slate-950 p-2.5 rounded-lg border border-slate-800 shrink-0 font-mono text-xs">
            <span className="text-slate-500">Pipeline Rule:</span>
            <span className="text-emerald-400 font-semibold">Strict Zero Fine-Tuning</span>
          </div>
        </div>
      </div>

      {/* Dataset Selection Cards & Degradation Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {GENERALIZATION_BENCHMARKS.map((bm) => {
          const isSelected = selectedDataset === bm.dataset;
          return (
            <button
              key={bm.dataset}
              onClick={() => setSelectedDataset(bm.dataset)}
              className={`p-4 rounded-xl border text-left transition-all relative ${
                isSelected
                  ? 'bg-cyan-950/30 border-cyan-500 ring-1 ring-cyan-500/50 shadow-lg'
                  : 'bg-slate-900 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-semibold text-cyan-400">
                  {bm.role}
                </span>
                <span className="text-[11px] font-mono text-slate-400">
                  N = {bm.sampleCount.toLocaleString()}
                </span>
              </div>
              <div className="text-base font-bold text-slate-100">{bm.displayName}</div>

              <div className="grid grid-cols-2 gap-2 mt-4 text-xs font-mono">
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500">ACCURACY</div>
                  <div className="text-sm font-bold text-slate-200 mt-0.5">
                    {(bm.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500">QWK KAPPA</div>
                  <div className="text-sm font-bold text-emerald-400 mt-0.5">
                    {bm.qwk.toFixed(3)}
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500">MACRO F1</div>
                  <div className="text-sm font-bold text-slate-200 mt-0.5">
                    {bm.macroF1.toFixed(3)}
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500">MEAN ENTROPY</div>
                  <div className="text-sm font-bold text-amber-400 mt-0.5">
                    {bm.meanEntropy.toFixed(3)} b
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Main Confusion Matrix & Clinical Misclassification Diagnostics */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Interactive 5x5 Confusion Matrix (7 cols) */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                5-Stage DR Confusion Matrix ({activeBenchmark.displayName})
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Rows: Ground Truth Clinical Stage · Columns: Predicted Stage
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowNormalized(false)}
                className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                  !showNormalized
                    ? 'bg-slate-800 text-cyan-300 font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Counts
              </button>
              <button
                onClick={() => setShowNormalized(true)}
                className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                  showNormalized
                    ? 'bg-slate-800 text-cyan-300 font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Normalized (%)
              </button>
            </div>
          </div>

          {/* Matrix Grid */}
          <div className="overflow-x-auto">
            <table className="w-full text-center border-collapse text-xs font-mono">
              <thead>
                <tr>
                  <th className="p-2 text-slate-500 text-left w-24">True \ Pred</th>
                  {[0, 1, 2, 3, 4].map((col) => (
                    <th key={col} className="p-2 text-slate-300 font-semibold">
                      {DR_STAGES[col as DRStage].shortName}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {currentMatrix.map((row, rIdx) => {
                  const rSum = rowSums[rIdx];
                  return (
                    <tr key={rIdx}>
                      <td className="p-2 text-left font-semibold text-slate-300 border-r border-slate-800">
                        {DR_STAGES[rIdx as DRStage].shortName}
                      </td>
                      {row.map((val, cIdx) => {
                        const isDiagonal = rIdx === cIdx;
                        const pct = rSum > 0 ? (val / rSum) * 100 : 0;
                        const intensity = isDiagonal ? pct / 100 : Math.min(1, (val / rSum) * 2.5);

                        return (
                          <td
                            key={cIdx}
                            onMouseEnter={() => setHoveredCell({ row: rIdx, col: cIdx, count: val })}
                            onMouseLeave={() => setHoveredCell(null)}
                            className="p-2 transition-colors cursor-pointer border border-slate-800/60"
                            style={{
                              backgroundColor: isDiagonal
                                ? `rgba(6, 182, 212, ${0.15 + intensity * 0.5})`
                                : val > 0
                                ? `rgba(239, 68, 68, ${intensity * 0.4})`
                                : 'transparent',
                            }}
                          >
                            <span
                              className={`font-semibold ${
                                isDiagonal ? 'text-cyan-200' : val > 0 ? 'text-rose-300' : 'text-slate-600'
                              }`}
                            >
                              {showNormalized ? `${pct.toFixed(1)}%` : val.toLocaleString()}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Hovered Cell Detail Strip */}
          <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono flex items-center justify-between text-slate-300 min-h-[38px]">
            {hoveredCell ? (
              <div className="flex items-center gap-4">
                <span>
                  True: <strong className="text-slate-100">{DR_STAGES[hoveredCell.row as DRStage].shortName}</strong>
                </span>
                <span>→</span>
                <span>
                  Predicted: <strong className="text-cyan-400">{DR_STAGES[hoveredCell.col as DRStage].shortName}</strong>
                </span>
                <span>·</span>
                <span>
                  Count: <strong className="text-amber-400">{hoveredCell.count.toLocaleString()} cases</strong>
                </span>
                <span>
                  ({((hoveredCell.count / rowSums[hoveredCell.row]) * 100).toFixed(1)}% of row)
                </span>
              </div>
            ) : (
              <span className="text-slate-500 italic">
                Hover over any matrix cell to inspect clinical transition details.
              </span>
            )}
          </div>
        </div>

        {/* Right: Domain Shift Analysis & Research Questions 5 & 6 (5 cols) */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-amber-400" />
              Empirical Findings on Domain Degradation
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Testing Hypotheses H3 (Performance drop) and H4 (Entropy surge)
            </p>
          </div>

          <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
              <div className="font-semibold text-slate-100 flex items-center justify-between">
                <span>1. Accuracy Decay Profile:</span>
                <span className="font-mono text-rose-400 font-bold">-10.5% (DDR)</span>
              </div>
              <p className="text-slate-400 text-[11px]">
                Internal APTOS accuracy of <strong>89.4%</strong> dropped to <strong>82.6%</strong> on IDRiD and <strong>78.9%</strong> on DDR. The drop was largely driven by subtle Mild NPDR cases being predicted as Normal due to differences in camera sensor optics (Zeiss vs Topcon).
              </p>
            </div>

            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
              <div className="font-semibold text-slate-100 flex items-center justify-between">
                <span>2. Predictive Uncertainty Shift:</span>
                <span className="font-mono text-amber-400 font-bold">+92.2% Entropy</span>
              </div>
              <p className="text-slate-400 text-[11px]">
                Crucially validating <strong>Hypothesis H4</strong>: As domain shift increased, mean predictive entropy escalated from 0.412 bits to 0.792 bits. This proves predictive entropy acts as a reliable sentinel for out-of-distribution difficulty.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
              <div className="font-semibold text-slate-100 flex items-center justify-between">
                <span>3. Quadratic Weighted Kappa (QWK):</span>
                <span className="font-mono text-emerald-400 font-bold">0.804 (Substantial)</span>
              </div>
              <p className="text-slate-400 text-[11px]">
                Because DR severity is ordinal, QWK penalizes severe mistakes (e.g. Stage 0 called Stage 4) heavily. QWK remained high across all three benchmarks (&gt; 0.80), demonstrating that model errors were strictly bounded to adjacent severity categories.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
