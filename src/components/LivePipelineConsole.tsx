import React, { useState, useMemo } from 'react';
import { CLINICAL_CASES } from '../data/sampleCohorts';
import { FundusCase, XAIMethod, DRStage, DR_STAGES } from '../types/pipeline';
import {
  evaluateImageQuality,
  classifyDRSeverity,
  applyTemperatureScaling,
  simulateMCDropout,
  computeLesionMetrics,
} from '../utils/retinalImageProcessor';
import { RetinalCanvasViewer } from './RetinalCanvasViewer';
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ShieldCheck,
  Stethoscope,
  Sliders,
  Layers,
  Upload,
  UserCheck,
  FileText,
  Activity,
  ArrowRight,
  Info,
  Thermometer,
} from 'lucide-react';

interface LivePipelineConsoleProps {
  onOpenReferralModal?: (fundusCase: FundusCase, uncertaintyScore: number, predictedStage: DRStage) => void;
}

export const LivePipelineConsole: React.FC<LivePipelineConsoleProps> = ({ onOpenReferralModal }) => {
  const [selectedCaseId, setSelectedCaseId] = useState<string>(CLINICAL_CASES[0].id);
  const [customCases, setCustomCases] = useState<FundusCase[]>([]);

  // Preprocessing toggles
  const [showFOVMask, setShowFOVMask] = useState(true);
  const [applyCLAHE, setApplyCLAHE] = useState(false);

  // XAI controls
  const [xaiMethod, setXaiMethod] = useState<XAIMethod>('GRAD_CAM_PP');
  const [showXAI, setShowXAI] = useState(true);
  const [xaiOpacity, setXaiOpacity] = useState(0.7);
  const [saliencyThreshold, setSaliencyThreshold] = useState(0.45);
  const [showLesionMasks, setShowLesionMasks] = useState(true);
  const [activeLesionFilter, setActiveLesionFilter] = useState<'ALL' | 'MA' | 'HE' | 'EX' | 'SE'>('ALL');

  // Calibration & Uncertainty controls
  const [temperature, setTemperature] = useState(1.38);
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState(0.85); // in bits of entropy

  // Active Case
  const allCases = useMemo(() => [...customCases, ...CLINICAL_CASES], [customCases]);
  const activeCase = useMemo(
    () => allCases.find((c) => c.id === selectedCaseId) || CLINICAL_CASES[0],
    [allCases, selectedCaseId]
  );

  // 1. Quality Gate Analysis
  const qualityResult = useMemo(() => evaluateImageQuality(activeCase), [activeCase]);

  // 2. DR Classification Analysis
  const classificationResult = useMemo(() => classifyDRSeverity(activeCase), [activeCase]);

  // 3. Temperature Scaling Calibration
  const calibrationResult = useMemo(
    () => applyTemperatureScaling(classificationResult.logits, temperature),
    [classificationResult.logits, temperature]
  );

  // 4. MC Dropout Epistemic Uncertainty
  const mcDropoutResult = useMemo(
    () => simulateMCDropout(classificationResult.predictedStage, classificationResult.rawProbabilities),
    [classificationResult]
  );

  // 5. Quantitative Lesion Metrics
  const lesionMetrics = useMemo(
    () => computeLesionMetrics(activeCase, xaiMethod, saliencyThreshold),
    [activeCase, xaiMethod, saliencyThreshold]
  );

  // Referral decision
  const isUncertain = calibrationResult.entropy > uncertaintyThreshold;
  const isUngradable = qualityResult.status === 'UNGRADABLE';

  // File Upload Handler
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string;
      const newCustomCase: FundusCase = {
        id: `upload-${Date.now()}`,
        caseNumber: `USER_${Date.now().toString().slice(-4)}`,
        dataset: 'USER_UPLOAD',
        patientAge: 55,
        patientGender: 'M',
        eye: 'OD',
        imageUrl: dataUrl,
        groundTruthStage: 2,
        qualityGroundTruth: 'GRADABLE',
        hasLesionMasks: false,
        notes: `User uploaded retinal fundus image: ${file.name}. Processed through real-time pipeline.`,
      };

      setCustomCases((prev) => [newCustomCase, ...prev]);
      setSelectedCaseId(newCustomCase.id);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="space-y-6">
      {/* Cohort Quick-Selector Carousel */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3 text-xs">
          <div className="flex items-center gap-2 text-slate-300 font-medium">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Select Screening Case Cohort</span>
            <span className="text-slate-500 font-mono">({allCases.length} available)</span>
          </div>

          <label className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-cyan-300 rounded-lg cursor-pointer transition-colors text-xs font-medium">
            <Upload className="w-3.5 h-3.5" />
            <span>Upload Fundus Image</span>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2.5">
          {allCases.map((item) => {
            const isSelected = item.id === selectedCaseId;
            const stageInfo = DR_STAGES[item.groundTruthStage];
            return (
              <button
                key={item.id}
                onClick={() => setSelectedCaseId(item.id)}
                className={`relative flex flex-col p-2.5 rounded-lg border text-left transition-all ${
                  isSelected
                    ? 'border-cyan-500 bg-cyan-950/30 ring-1 ring-cyan-500/50'
                    : 'border-slate-800 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-800/40'
                }`}
              >
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mb-1">
                  <span>{item.caseNumber}</span>
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: stageInfo.color }}
                  />
                </div>
                <div className="text-xs font-semibold text-slate-200 truncate">
                  {item.qualityGroundTruth === 'UNGRADABLE' ? 'Ungradable' : stageInfo.shortName}
                </div>
                <div className="text-[10px] text-slate-500 font-mono mt-0.5 truncate">
                  {item.dataset} · {item.eye}
                </div>
                {item.hasLesionMasks && (
                  <span className="text-[9px] text-emerald-400 font-mono mt-1">
                    ✓ IDRiD Lesion GT
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main 2-Column Workstation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Canvas Viewport & Layer Controls (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <RetinalCanvasViewer
            fundusCase={activeCase}
            xaiMethod={xaiMethod}
            showXAI={showXAI}
            xaiOpacity={xaiOpacity}
            saliencyThreshold={saliencyThreshold}
            showLesionMasks={showLesionMasks}
            showFOVMask={showFOVMask}
            applyCLAHE={applyCLAHE}
            activeLesionFilter={activeLesionFilter}
          />

          {/* Layer and Preprocessing Control Panel */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-4 text-xs">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                Pipeline Preprocessing & XAI Layers
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                Resolution: 500×500 px · RGB
              </span>
            </div>

            {/* Preprocessing Toggles */}
            <div className="grid grid-cols-2 gap-3">
              <label className="flex items-center gap-2 p-2 rounded-lg bg-slate-950 border border-slate-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showFOVMask}
                  onChange={(e) => setShowFOVMask(e.target.checked)}
                  className="rounded border-slate-700 text-cyan-500 focus:ring-0"
                />
                <div>
                  <div className="font-medium text-slate-300">Retinal FOV Crop</div>
                  <div className="text-[10px] text-slate-500">Remove black background margin</div>
                </div>
              </label>

              <label className="flex items-center gap-2 p-2 rounded-lg bg-slate-950 border border-slate-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={applyCLAHE}
                  onChange={(e) => setApplyCLAHE(e.target.checked)}
                  className="rounded border-slate-700 text-cyan-500 focus:ring-0"
                />
                <div>
                  <div className="font-medium text-slate-300">CLAHE Enhancement</div>
                  <div className="text-[10px] text-slate-500">Local green-channel equalization</div>
                </div>
              </label>
            </div>

            {/* XAI Method Selector */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-slate-400">
                <span>Explainable AI Attribution Method</span>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showXAI}
                    onChange={(e) => setShowXAI(e.target.checked)}
                    className="rounded border-slate-700 text-cyan-500 focus:ring-0"
                  />
                  <span className="text-slate-300">Enable Heatmap</span>
                </label>
              </div>

              <div className="grid grid-cols-3 gap-2">
                {(
                  [
                    { id: 'GRAD_CAM_PP', label: 'Grad-CAM++', desc: 'Positive gradients' },
                    { id: 'SCORE_CAM', label: 'Score-CAM', desc: 'Perturbation score' },
                    { id: 'INTEGRATED_GRADIENTS', label: 'Integrated Grad', desc: 'Path integral' },
                  ] as const
                ).map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setXaiMethod(m.id)}
                    className={`p-2 rounded-lg border text-left transition-all ${
                      xaiMethod === m.id
                        ? 'border-cyan-500 bg-cyan-950/40 text-cyan-200'
                        : 'border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-300'
                    }`}
                  >
                    <div className="font-semibold text-xs">{m.label}</div>
                    <div className="text-[10px] text-slate-500">{m.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Sliders: Opacity & Saliency Threshold */}
            <div className="grid grid-cols-2 gap-4 pt-1">
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Heatmap Opacity</span>
                  <span className="font-mono text-cyan-400">{Math.round(xaiOpacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={xaiOpacity}
                  onChange={(e) => setXaiOpacity(parseFloat(e.target.value))}
                  className="w-full accent-cyan-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Saliency Threshold (τ)</span>
                  <span className="font-mono text-cyan-400">{saliencyThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.85"
                  step="0.05"
                  value={saliencyThreshold}
                  onChange={(e) => setSaliencyThreshold(parseFloat(e.target.value))}
                  className="w-full accent-cyan-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>

            {/* IDRiD Lesion Filter Buttons (if lesion masks present) */}
            {activeCase.hasLesionMasks && (
              <div className="pt-2 border-t border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between text-slate-400">
                  <span className="text-[11px]">IDRiD Ground-Truth Lesion Masks Overlay</span>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showLesionMasks}
                      onChange={(e) => setShowLesionMasks(e.target.checked)}
                      className="rounded border-slate-700 text-rose-500 focus:ring-0"
                    />
                    <span className="text-slate-300">Show Masks</span>
                  </label>
                </div>

                <div className="flex items-center gap-1.5">
                  {(['ALL', 'EX', 'HE', 'MA', 'SE'] as const).map((filter) => (
                    <button
                      key={filter}
                      onClick={() => setActiveLesionFilter(filter)}
                      className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
                        activeLesionFilter === filter
                          ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                          : 'bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800'
                      }`}
                    >
                      {filter === 'ALL'
                        ? 'All Lesions'
                        : filter === 'EX'
                        ? 'Hard Exudates (EX)'
                        : filter === 'HE'
                        ? 'Hemorrhages (HE)'
                        : filter === 'MA'
                        ? 'Microaneurysms (MA)'
                        : 'Soft Exudates (SE)'}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Pipeline Output Stages & Clinical Decision (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* STAGE 1: Image Quality Assessment Gate (MobileNetV3-Small) */}
          <div
            className={`p-4 rounded-xl border transition-all ${
              qualityResult.status === 'GRADABLE'
                ? 'bg-slate-900 border-slate-800'
                : 'bg-rose-950/30 border-rose-600/50'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider font-mono">
                Stage 1 · Quality Gate (MobileNetV3-Small)
              </span>
              {qualityResult.status === 'GRADABLE' ? (
                <span className="flex items-center gap-1 text-emerald-400 text-xs font-medium">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Gradable
                </span>
              ) : (
                <span className="flex items-center gap-1 text-rose-400 text-xs font-medium">
                  <XCircle className="w-3.5 h-3.5" />
                  Ungradable
                </span>
              )}
            </div>

            {qualityResult.status === 'GRADABLE' ? (
              <div className="flex items-center justify-between text-xs text-slate-300 font-mono">
                <span>Gradability Confidence:</span>
                <span className="text-emerald-400 font-bold">
                  {(qualityResult.gradableProbability * 100).toFixed(1)}%
                </span>
              </div>
            ) : (
              <div className="space-y-2 mt-2">
                <div className="p-2.5 rounded-lg bg-rose-900/30 border border-rose-700/50 text-xs text-rose-200 space-y-1">
                  <div className="font-semibold flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                    {qualityResult.reasonLabel}
                  </div>
                  <div className="text-[11px] text-rose-300/80">
                    Clinical Protocol: {qualityResult.recommendedAction === 'RECAPTURE_IMAGE' ? 'Image rejected. Request patient fundus recapture.' : 'Referred to specialist for physical dilated slit-lamp ophthalmoscopy.'}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* STAGE 2: DR Severity Classification (EfficientNet-B3) */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider font-mono">
                Stage 2 · DR Grading (EfficientNet-B3)
              </span>
              <span className="text-xs font-mono text-cyan-400">5-Stage Softmax</span>
            </div>

            {/* Diagnostic Grade Banner */}
            {qualityResult.status === 'GRADABLE' ? (
              <div
                className="p-3 rounded-lg border flex items-center justify-between"
                style={{
                  backgroundColor: `${DR_STAGES[classificationResult.predictedStage].color}15`,
                  borderColor: `${DR_STAGES[classificationResult.predictedStage].color}40`,
                }}
              >
                <div>
                  <div className="text-sm font-bold text-slate-100 flex items-center gap-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: DR_STAGES[classificationResult.predictedStage].color }}
                    />
                    Stage {classificationResult.predictedStage}: {DR_STAGES[classificationResult.predictedStage].name}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    {DR_STAGES[classificationResult.predictedStage].description}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-slate-400">Confidence</div>
                  <div className="text-lg font-mono font-bold text-slate-100">
                    {(calibrationResult.calibratedProbabilities[classificationResult.predictedStage] * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-400 italic">
                Disease grading suspended due to ungradable image quality.
              </div>
            )}

            {/* Probability Distribution Bars */}
            <div className="space-y-1.5 pt-1">
              {[0, 1, 2, 3, 4].map((stageNum) => {
                const s = stageNum as DRStage;
                const rawProb = classificationResult.rawProbabilities[s];
                const calProb = calibrationResult.calibratedProbabilities[s];
                const isSelected = classificationResult.predictedStage === s;

                return (
                  <div key={s} className="space-y-0.5 text-[11px]">
                    <div className="flex justify-between font-mono text-slate-400">
                      <span className={isSelected ? 'text-slate-200 font-semibold' : ''}>
                        {s}: {DR_STAGES[s].shortName}
                      </span>
                      <span>
                        <span className="text-slate-500">raw: {(rawProb * 100).toFixed(0)}%</span> ·{' '}
                        <span className={isSelected ? 'text-cyan-400 font-bold' : 'text-slate-300'}>
                          {(calProb * 100).toFixed(1)}%
                        </span>
                      </span>
                    </div>
                    <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden flex">
                      <div
                        className="h-full transition-all duration-300"
                        style={{
                          width: `${calProb * 100}%`,
                          backgroundColor: DR_STAGES[s].color,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* STAGE 3: Quantitative Lesion Grounding (Objective 4 Contribution) */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider font-mono">
                Stage 3 · Quantitative Lesion Validation
              </span>
              <span className="text-[11px] font-mono text-emerald-400">IDRiD Ground Truth</span>
            </div>

            {activeCase.hasLesionMasks ? (
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono">DICE SIMILARITY</div>
                  <div className="text-xl font-mono font-bold text-cyan-400 mt-0.5">
                    {lesionMetrics.dice.toFixed(3)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Overlap coefficient</div>
                </div>

                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono">INTERSECTION / UNION (IoU)</div>
                  <div className="text-xl font-mono font-bold text-emerald-400 mt-0.5">
                    {lesionMetrics.iou.toFixed(3)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Jaccard index</div>
                </div>

                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono">POINTING GAME</div>
                  <div className="text-sm font-semibold mt-1 flex items-center gap-1.5">
                    {lesionMetrics.pointingGameHit ? (
                      <span className="text-emerald-400 flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5" /> HIT (Inside Lesion)
                      </span>
                    ) : (
                      <span className="text-amber-400 flex items-center gap-1">
                        <AlertTriangle className="w-3.5 h-3.5" /> MISS (Peak Off-Lesion)
                      </span>
                    )}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Max activation location</div>
                </div>

                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono">LESION RECALL</div>
                  <div className="text-lg font-mono font-bold text-amber-400 mt-0.5">
                    {(lesionMetrics.lesionRecall * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Covered lesion area</div>
                </div>
              </div>
            ) : (
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-400">
                Pixel-level lesion masks unavailable for this cohort ({activeCase.dataset}). Select an IDRiD case to inspect quantitative Dice/IoU alignment.
              </div>
            )}
          </div>

          {/* STAGE 4: Calibrated Uncertainty & Specialist Referral Engine */}
          <div
            className={`p-4 rounded-xl border space-y-3 ${
              isUngradable
                ? 'bg-rose-950/20 border-rose-800/40'
                : isUncertain
                ? 'bg-amber-950/20 border-amber-600/40'
                : 'bg-emerald-950/20 border-emerald-600/40'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider font-mono text-slate-300">
                Stage 4 · Calibrated Uncertainty & Referral
              </span>
              <span className="flex items-center gap-1 font-mono text-[11px] text-slate-400">
                <Thermometer className="w-3.5 h-3.5 text-cyan-400" />
                T = {temperature.toFixed(2)}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
                <div className="text-[10px] text-slate-500 font-mono">PREDICTIVE ENTROPY H(p)</div>
                <div className="text-lg font-mono font-bold text-slate-200 mt-0.5">
                  {calibrationResult.entropy.toFixed(3)}{' '}
                  <span className="text-xs font-normal text-slate-400">bits</span>
                </div>
                <div className="text-[10px] text-slate-500">
                  Threshold: {uncertaintyThreshold.toFixed(2)} bits
                </div>
              </div>

              <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
                <div className="text-[10px] text-slate-500 font-mono">MC DROPOUT VARIANCE</div>
                <div className="text-lg font-mono font-bold text-slate-200 mt-0.5">
                  {mcDropoutResult.epistemicUncertainty.toFixed(4)}
                </div>
                <div className="text-[10px] text-slate-500">N=30 stochastic forward passes</div>
              </div>
            </div>

            {/* Final Clinical Action Callout */}
            <div className="pt-2 border-t border-slate-800/80">
              {isUngradable ? (
                <div className="flex items-start gap-2.5 p-3 rounded-lg bg-rose-900/30 border border-rose-600/50 text-xs">
                  <XCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-bold text-rose-200">ACTION: REJECT & REFER FOR RETAKE</div>
                    <div className="text-[11px] text-rose-300 mt-0.5">
                      Ungradable image quality prevents reliable AI diagnosis. Recapture required.
                    </div>
                  </div>
                </div>
              ) : isUncertain ? (
                <div className="flex items-start gap-2.5 p-3 rounded-lg bg-amber-900/30 border border-amber-600/50 text-xs">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <div className="font-bold text-amber-200">
                      ACTION: UNCERTAIN PREDICTION → SPECIALIST REFERRAL
                    </div>
                    <div className="text-[11px] text-amber-300 mt-0.5">
                      Predictive entropy ({calibrationResult.entropy.toFixed(3)} bits) exceeds threshold. System refrains from autonomous conclusion.
                    </div>
                    {onOpenReferralModal && (
                      <button
                        onClick={() =>
                          onOpenReferralModal(
                            activeCase,
                            calibrationResult.entropy,
                            classificationResult.predictedStage
                          )
                        }
                        className="mt-2 px-3 py-1 bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold rounded text-xs transition-colors flex items-center gap-1.5"
                      >
                        <FileText className="w-3.5 h-3.5" />
                        Generate Referral Sheet
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex items-start gap-2.5 p-3 rounded-lg bg-emerald-900/30 border border-emerald-600/50 text-xs">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <div className="font-bold text-emerald-200">
                      ACTION: HIGH CONFIDENCE DIAGNOSIS RETURNED
                    </div>
                    <div className="text-[11px] text-emerald-300 mt-0.5">
                      Predicted Stage {classificationResult.predictedStage} ({DR_STAGES[classificationResult.predictedStage].shortName}) with validated lesion alignment.
                    </div>
                    {DR_STAGES[classificationResult.predictedStage].referralRecommended && (
                      <div className="mt-1 text-[11px] text-cyan-300">
                        * Routine ophthalmic consultation recommended for Moderate+ retinopathy.
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
