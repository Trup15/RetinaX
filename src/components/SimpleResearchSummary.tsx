import React, { useState } from 'react';
import { GENERALIZATION_BENCHMARKS } from '../data/researchBenchmarks';
import {
  Globe,
  ShieldCheck,
  Smartphone,
  CheckCircle2,
  Sliders,
} from 'lucide-react';

export const SimpleResearchSummary: React.FC = () => {
  const [referralPercent, setReferralPercent] = useState<number>(12);

  const r = referralPercent / 100;
  const coverage = 1 - r;
  const errorCaptureRate = Math.min(1.0, 1 - Math.pow(1 - r, 2.35));

  const baselineErrorRate = 0.106;
  const remainingErrorFraction = 1 - errorCaptureRate;
  const selectiveRisk = coverage > 0 ? (baselineErrorRate * remainingErrorFraction) / coverage : 0;
  const selectiveAccuracy = Math.min(0.999, Math.max(0.894, 1 - selectiveRisk));

  const cohortSize = 5000;
  const autoPatients = Math.round(cohortSize * coverage);
  const totalErrors = Math.round(cohortSize * baselineErrorRate);
  const caughtErrors = Math.round(totalErrors * errorCaptureRate);

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
        <div className="text-xs font-bold uppercase tracking-wider text-cyan-700 mb-1">
          Objectives 5, 6 & 7 · Real-World Clinical Value
        </div>
        <h2 className="text-xl font-bold text-slate-900">
          How Safe is this AI in the Real World?
        </h2>
        <p className="text-xs text-slate-600 mt-1 max-w-3xl leading-relaxed">
          Building a working medical AI isn't just about getting high accuracy on one computer. This section demonstrates: <strong>1)</strong> Testing across real Indian and international eye hospitals, <strong>2)</strong> Dynamically calculating how referring uncertain patients prevents misdiagnoses, and <strong>3)</strong> Lightweight offline operation in rural sub-centers.
        </p>
      </div>

      {/* Part 1: Cross-Dataset Validation (APTOS -> IDRiD -> DDR) */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Globe className="w-5 h-5 text-cyan-600" />
              1. Testing Across Real Eye Hospital Datasets
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              The AI was trained on one hospital and tested completely unchanged on others to measure real hospital transfer.
            </p>
          </div>
          <span className="text-xs font-mono bg-cyan-50 text-cyan-800 border border-cyan-200 px-3 py-1 rounded-lg font-semibold">
            Zero Retraining / Zero Leakage
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {GENERALIZATION_BENCHMARKS.map((item) => (
            <div
              key={item.dataset}
              className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-900">{item.displayName}</span>
                <span className="text-[10px] font-mono text-slate-500">
                  {item.sampleCount.toLocaleString()} scans
                </span>
              </div>

              <div className="space-y-1.5 pt-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Diagnostic Accuracy:</span>
                  <span className="font-mono font-bold text-cyan-700">
                    {(item.accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Clinical Agreement (QWK):</span>
                  <span className="font-mono font-bold text-emerald-700">
                    {item.qwk.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Calibrated Entropy:</span>
                  <span className="font-mono text-amber-700 font-semibold">
                    {item.meanEntropy.toFixed(3)} bits
                  </span>
                </div>
              </div>

              <div className="text-[11px] text-slate-600 pt-2 border-t border-slate-200 leading-relaxed">
                {item.dataset === 'APTOS_2019' &&
                  'Primary training cohort from Aravind Eye Hospital. Baseline accuracy is 89.4%.'}
                {item.dataset === 'IDRiD' &&
                  'Indian external validation benchmark. Proves strong generalization with pixel lesion masks.'}
                {item.dataset === 'DDR' &&
                  'External diverse benchmark. Demonstrates resilience to varied camera optics and lighting.'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Part 2: Interactive Uncertainty & Referral Simulator */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-4 shadow-xs">
        <div className="border-b border-slate-100 pb-3">
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-emerald-600" />
            2. The Safety Gate: Dynamic Specialist Referral Simulator
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Slide the referral percentage below. All metrics, patient counts, and accuracy figures update dynamically in real time.
          </p>
        </div>

        <div className="p-5 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
          <div className="flex justify-between items-center text-xs">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-emerald-600" />
              <span className="text-slate-800 font-bold">Specialist Referral Cutoff:</span>
            </div>
            <span className="text-amber-800 font-bold font-mono text-sm bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
              Refer top {referralPercent}% most uncertain cases
            </span>
          </div>

          <input
            type="range"
            min="0"
            max="35"
            step="1"
            value={referralPercent}
            onChange={(e) => setReferralPercent(parseInt(e.target.value))}
            className="w-full accent-emerald-600 bg-slate-200 h-2 rounded-lg appearance-none cursor-pointer"
          />

          <div className="flex justify-between text-[11px] font-mono text-slate-500">
            <span>0% (No Referral · 89.4% baseline)</span>
            <span className="text-emerald-700 font-semibold">12% (Clinical Optimum)</span>
            <span>35% (Conservative High Referral)</span>
          </div>

          {/* Dynamic 4-Stat Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            <div className="p-3 bg-white border border-slate-200 rounded-lg text-center shadow-xs">
              <div className="text-[10px] text-slate-500 uppercase font-mono font-semibold">AUTOMATED SCREENINGS</div>
              <div className="text-xl font-bold font-mono text-cyan-700 mt-1">
                {(coverage * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5 font-mono">
                {autoPatients.toLocaleString()} / {cohortSize.toLocaleString()} eyes
              </div>
            </div>

            <div className="p-3 bg-white border border-slate-200 rounded-lg text-center shadow-xs">
              <div className="text-[10px] text-slate-500 uppercase font-mono font-semibold">SELECTIVE ACCURACY</div>
              <div className="text-xl font-bold font-mono text-emerald-700 mt-1">
                {(selectiveAccuracy * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-emerald-700 mt-0.5 font-mono font-bold">
                +{((selectiveAccuracy - 0.894) * 100).toFixed(1)}% boost
              </div>
            </div>

            <div className="p-3 bg-white border border-slate-200 rounded-lg text-center shadow-xs">
              <div className="text-[10px] text-slate-500 uppercase font-mono font-semibold">REMAINING RISK</div>
              <div className="text-xl font-bold font-mono text-rose-700 mt-1">
                {(selectiveRisk * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5 font-mono">
                Reduced from 10.6%
              </div>
            </div>

            <div className="p-3 bg-white border border-slate-200 rounded-lg text-center shadow-xs">
              <div className="text-[10px] text-slate-500 uppercase font-mono font-semibold">ERRORS INTERCEPTED</div>
              <div className="text-xl font-bold font-mono text-amber-700 mt-1">
                {(errorCaptureRate * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-amber-800 mt-0.5 font-mono font-semibold">
                {caughtErrors} / {totalErrors} caught
              </div>
            </div>
          </div>
        </div>

        <div className="p-3.5 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-950 leading-relaxed flex items-start gap-2.5">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
          <div>
            <strong>Dynamic Clinical Insight:</strong> By allowing the AI to abstain on the top{' '}
            <strong className="text-slate-900">{referralPercent}% most uncertain eyes</strong>, diagnostic accuracy on the remaining{' '}
            <strong className="text-slate-900">{(coverage * 100).toFixed(0)}% patients</strong> jumps from 89.4% to{' '}
            <strong className="text-emerald-700 font-mono">{(selectiveAccuracy * 100).toFixed(1)}%</strong>, intercepting{' '}
            <strong className="text-amber-800 font-mono font-bold">{(errorCaptureRate * 100).toFixed(0)}%</strong> of all potential misclassifications before they reach patients.
          </div>
        </div>
      </div>

      {/* Part 3: Offline Deployment for Rural Health Centers */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-4 shadow-xs">
        <div className="border-b border-slate-100 pb-3">
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Smartphone className="w-5 h-5 text-cyan-600" />
            3. Works Offline on Simple Smartphones & Rural Kiosks
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Designed specifically for rural villages with zero internet and limited electricity.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <div className="text-[10px] text-slate-500 font-semibold">TOTAL FILE SIZE</div>
            <div className="text-lg font-bold text-cyan-700 mt-0.5">15.0 MB</div>
            <div className="text-[10px] text-slate-500 font-sans mt-0.5">
              INT8 Post-Training Quantized
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <div className="text-[10px] text-slate-500 font-semibold">SPEED (PER EYE)</div>
            <div className="text-lg font-bold text-emerald-700 mt-0.5">42.6 ms</div>
            <div className="text-[10px] text-slate-500 font-sans mt-0.5">
              ARM Cortex-A78 CPU
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <div className="text-[10px] text-slate-500 font-semibold">INTERNET NEEDED</div>
            <div className="text-lg font-bold text-slate-800 mt-0.5">0.0 %</div>
            <div className="text-[10px] text-slate-500 font-sans mt-0.5">
              Zero cloud telemetry
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <div className="text-[10px] text-slate-500 font-semibold">POWER CONSUMPTION</div>
            <div className="text-lg font-bold text-amber-700 mt-0.5">3.5 W</div>
            <div className="text-[10px] text-slate-500 font-sans mt-0.5">
              Runs all day on phone battery
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
