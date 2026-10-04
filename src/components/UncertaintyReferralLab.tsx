import React, { useState } from 'react';
import {
  CALIBRATION_BINS,
  RISK_COVERAGE_CURVE,
} from '../data/researchBenchmarks';
import {
  Thermometer,
  ShieldCheck,
  TrendingUp,
  UserCheck,
  AlertTriangle,
  Sliders,
  CheckCircle2,
  PieChart,
} from 'lucide-react';

export const UncertaintyReferralLab: React.FC = () => {
  const [temperature, setTemperature] = useState(1.38);
  const [selectedReferralIndex, setSelectedReferralIndex] = useState(3); // 15% referral rate by default

  const currentCurvePoint = RISK_COVERAGE_CURVE[selectedReferralIndex];

  return (
    <div className="space-y-6">
      {/* Context Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="text-xs font-mono text-cyan-400 font-semibold uppercase tracking-wider mb-1">
              Objective 6 · Trustworthy Decision Support
            </div>
            <h2 className="text-xl font-bold text-slate-100">
              Confidence Calibration & Selective Specialist Referral
            </h2>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
              Addressing <strong className="text-slate-300">Research Gap 4 (RG4)</strong>: Standard neural networks output overconfident uncalibrated probabilities. By learning a validation-fitted Temperature Scaling parameter (<strong className="text-cyan-300">T = 1.38</strong>) and establishing an uncertainty-based abstention mechanism, the system refrains from diagnosing difficult ambiguous eyes and selectively routes them to ophthalmologists.
            </p>
          </div>

          <div className="flex items-center gap-3 bg-slate-950 p-3 rounded-lg border border-slate-800 shrink-0 font-mono text-xs">
            <ShieldCheck className="w-8 h-8 text-emerald-400" />
            <div>
              <div className="text-[10px] text-slate-500">SELECTIVE ACCURACY</div>
              <div className="text-lg font-bold text-emerald-400">
                {(currentCurvePoint.selectiveAccuracy * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-slate-400">
                on {(currentCurvePoint.coverage * 100).toFixed(0)}% retained cohort
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top 3 Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>ECE PRE-CALIBRATION</span>
            <span className="text-rose-400 font-bold">8.9% Error</span>
          </div>
          <div className="text-2xl font-mono font-bold text-slate-200">0.089</div>
          <p className="text-[11px] text-slate-500">
            Raw softmax probabilities overstate true diagnostic likelihood.
          </p>
        </div>

        <div className="bg-slate-900 border-2 border-emerald-500/40 rounded-xl p-4 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>ECE POST-CALIBRATION (T=1.38)</span>
            <span className="text-emerald-400 font-bold">57% Reduction</span>
          </div>
          <div className="text-2xl font-mono font-bold text-emerald-400">0.038</div>
          <p className="text-[11px] text-slate-500">
            Expected Calibration Error reduced to clinically acceptable bound.
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>ERROR CAPTURE RATE</span>
            <span className="text-amber-400 font-bold">Target: Difficult Eyes</span>
          </div>
          <div className="text-2xl font-mono font-bold text-amber-400">
            {(currentCurvePoint.errorCaptureRate * 100).toFixed(0)}%
          </div>
          <p className="text-[11px] text-slate-500">
            Of all model errors are intercepted before reaching the patient.
          </p>
        </div>
      </div>

      {/* Main Interactive Sections: Reliability Diagram & Risk-Coverage Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: 10-Bin Reliability Diagram (6 cols) */}
        <div className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <Thermometer className="w-4 h-4 text-cyan-400" />
                Reliability Diagram (Confidence Calibration)
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Expected Accuracy vs Mean Confidence (10 Bins)
              </p>
            </div>
            <span className="text-xs font-mono text-cyan-400 bg-slate-950 px-2 py-1 rounded border border-slate-800">
              T = {temperature.toFixed(2)}
            </span>
          </div>

          {/* Bar Diagram */}
          <div className="h-60 bg-slate-950 border border-slate-800 rounded-lg p-3 relative flex items-end justify-between gap-1">
            {CALIBRATION_BINS.map((bin, i) => {
              const confHeight = bin.avgConf * 100;
              const preAccHeight = bin.preAcc * 100;
              const postAccHeight = bin.postAcc * 100;

              return (
                <div key={i} className="flex-1 flex flex-col items-center h-full justify-end group relative">
                  {/* Perfect calibration reference line tick */}
                  <div
                    className="absolute w-full border-t border-slate-600 border-dashed z-10"
                    style={{ bottom: `${confHeight}%` }}
                  />

                  {/* Pre-calibration Bar (Grey/Rose) */}
                  <div
                    className="w-full bg-slate-700/60 rounded-t transition-all group-hover:bg-slate-600"
                    style={{ height: `${preAccHeight}%` }}
                  />

                  {/* Post-calibration Bar (Cyan) */}
                  <div
                    className="w-full bg-cyan-500/80 rounded-t -mt-full transition-all group-hover:bg-cyan-400"
                    style={{ height: `${postAccHeight}%` }}
                  />

                  {/* Tooltip on hover */}
                  <div className="absolute bottom-full mb-2 hidden group-hover:flex flex-col bg-slate-900 border border-slate-700 p-2 rounded shadow-xl text-[10px] font-mono z-30 pointer-events-none w-32">
                    <span className="text-slate-400">Bin: {bin.bin}</span>
                    <span className="text-slate-300">Avg Conf: {(bin.avgConf * 100).toFixed(0)}%</span>
                    <span className="text-rose-400">Pre Acc: {(bin.preAcc * 100).toFixed(0)}%</span>
                    <span className="text-cyan-400 font-bold">Post Acc: {(bin.postAcc * 100).toFixed(0)}%</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 px-1">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-slate-700 rounded-sm" /> Pre-Scaling
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-cyan-500 rounded-sm" /> Post-Scaling (T=1.38)
              </span>
              <span className="flex items-center gap-1 text-slate-500">
                --- Ideal Calibration
              </span>
            </div>
            <span>Bin Width = 0.1</span>
          </div>

          {/* Temperature Scrubber */}
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 space-y-1.5">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">Temperature Scaling Parameter (T):</span>
              <span className="text-cyan-400 font-bold">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.5"
              step="0.02"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-cyan-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>0.50 (Overconfident)</span>
              <span className="text-cyan-400">1.38 (Optimal Validation Fit)</span>
              <span>2.50 (Underconfident)</span>
            </div>
          </div>
        </div>

        {/* Right: Risk-Coverage Curve & Selective Referral Engine (6 cols) */}
        <div className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                Risk-Coverage Curve (Selective Referral)
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Balancing autonomous clinic throughput against specialist referral safety
              </p>
            </div>
            <span className="text-xs font-mono text-emerald-400 bg-slate-950 px-2 py-1 rounded border border-slate-800">
              Coverage: {(currentCurvePoint.coverage * 100).toFixed(0)}%
            </span>
          </div>

          {/* Interactive Curve Scrubber */}
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 space-y-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-300 font-semibold">Specialist Referral Cutoff:</span>
              <span className="text-amber-400 font-bold">
                {(currentCurvePoint.referralRate * 100).toFixed(0)}% Referred
              </span>
            </div>

            <input
              type="range"
              min="0"
              max={RISK_COVERAGE_CURVE.length - 1}
              step="1"
              value={selectedReferralIndex}
              onChange={(e) => setSelectedReferralIndex(parseInt(e.target.value))}
              className="w-full accent-emerald-500 bg-slate-800 h-2 rounded-lg appearance-none cursor-pointer"
            />

            <div className="grid grid-cols-3 gap-2 pt-2 text-xs font-mono">
              <div className="p-2 rounded bg-slate-900 border border-slate-800">
                <div className="text-[10px] text-slate-500">COVERAGE</div>
                <div className="text-sm font-bold text-slate-200 mt-0.5">
                  {(currentCurvePoint.coverage * 100).toFixed(0)}%
                </div>
                <div className="text-[9px] text-slate-500">Auto-screened</div>
              </div>

              <div className="p-2 rounded bg-slate-900 border border-slate-800">
                <div className="text-[10px] text-slate-500">SELECTIVE ACCURACY</div>
                <div className="text-sm font-bold text-emerald-400 mt-0.5">
                  {(currentCurvePoint.selectiveAccuracy * 100).toFixed(1)}%
                </div>
                <div className="text-[9px] text-slate-500">Retained cohort</div>
              </div>

              <div className="p-2 rounded bg-slate-900 border border-slate-800">
                <div className="text-[10px] text-slate-500">SELECTIVE RISK</div>
                <div className="text-sm font-bold text-rose-400 mt-0.5">
                  {(currentCurvePoint.selectiveRisk * 100).toFixed(1)}%
                </div>
                <div className="text-[9px] text-slate-500">Remaining error</div>
              </div>
            </div>
          </div>

          {/* Clinical Safety Takeaway */}
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-300 space-y-2 leading-relaxed">
            <div className="font-semibold text-emerald-400 flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Hypothesis 7 Validated:
            </div>
            <p>
              By permitting the deep learning system to <strong className="text-slate-100">abstain on the top {(currentCurvePoint.referralRate * 100).toFixed(0)}% most uncertain cases</strong>, diagnostic accuracy on the remaining {(currentCurvePoint.coverage * 100).toFixed(0)}% patients jumps from 89.4% to <strong className="text-emerald-400">{(currentCurvePoint.selectiveAccuracy * 100).toFixed(1)}%</strong>.
            </p>
            <p className="text-slate-400 text-[11px]">
              This captures <strong className="text-amber-300">{(currentCurvePoint.errorCaptureRate * 100).toFixed(0)}% of all system errors</strong>, ensuring severe or boundary proliferative retinopathy cases are reviewed directly by ophthalmologists.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
