import React from 'react';
import { FundusCase, DRStage, DR_STAGES } from '../types/pipeline';
import { X, Printer, Stethoscope, AlertTriangle, ShieldCheck } from 'lucide-react';

interface ReferralModalProps {
  fundusCase: FundusCase;
  uncertaintyScore: number;
  predictedStage: DRStage;
  onClose: () => void;
}

export const ReferralModal: React.FC<ReferralModalProps> = ({
  fundusCase,
  uncertaintyScore,
  predictedStage,
  onClose,
}) => {
  const handlePrint = () => {
    window.print();
  };

  const isUngradable = fundusCase.qualityGroundTruth === 'UNGRADABLE';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs overflow-y-auto">
      <div className="bg-white border border-slate-200 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl my-8 animate-in fade-in zoom-in-95 duration-200 text-slate-900">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-slate-50 border-b border-slate-200">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-amber-100 text-amber-800 border border-amber-200">
              <Stethoscope className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Ophthalmic Clinical Specialist Referral Sheet
              </h3>
              <p className="text-xs text-slate-500 font-mono">
                Protocol: Calibrated Uncertainty-Aware Triage (T = 1.38)
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-200/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body / Printable Document */}
        <div className="p-6 space-y-5 text-slate-800 text-xs">
          {/* Patient Demographics Banner */}
          <div className="grid grid-cols-4 gap-3 p-3.5 rounded-xl bg-slate-50 border border-slate-200 font-mono">
            <div>
              <div className="text-[10px] text-slate-500 font-semibold">PATIENT ID</div>
              <div className="text-sm font-bold text-slate-900 mt-0.5">{fundusCase.caseNumber}</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500 font-semibold">AGE / GENDER</div>
              <div className="text-sm font-bold text-slate-900 mt-0.5">
                {fundusCase.patientAge} Yrs / {fundusCase.patientGender}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500 font-semibold">EXAMINED EYE</div>
              <div className="text-sm font-bold text-cyan-700 mt-0.5">
                {fundusCase.eye === 'OD' ? 'Right Eye (OD)' : 'Left Eye (OS)'}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500 font-semibold">SOURCE COHORT</div>
              <div className="text-sm font-bold text-slate-900 mt-0.5">{fundusCase.dataset}</div>
            </div>
          </div>

          {/* Referral Cause Alert */}
          <div
            className={`p-4 rounded-xl border flex items-start gap-3 ${
              isUngradable
                ? 'bg-rose-50 border-rose-200 text-rose-950'
                : 'bg-amber-50 border-amber-200 text-amber-950'
            }`}
          >
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-bold text-sm">
                {isUngradable
                  ? 'Referral Trigger: Optical Ungradability (MobileNetV3 Quality Gate)'
                  : 'Referral Trigger: Predictive Uncertainty Exceeded (H(p) > 0.85 bits)'}
              </div>
              <p className="text-xs leading-relaxed opacity-90">
                {isUngradable
                  ? 'Severe optical artifacts or media opacities preclude automated diabetic retinopathy grading. Direct slit-lamp examination required.'
                  : `Model predictive entropy (${uncertaintyScore.toFixed(3)} bits) falls within the abstention envelope. Autonomous diagnosis suspended to preserve clinical safety.`}
              </p>
            </div>
          </div>

          {/* Clinical Findings Comparison */}
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-[11px] font-mono text-slate-500 font-bold uppercase">
                Image Quality Assessment
              </div>
              <div className="flex items-center justify-between">
                <span>Status:</span>
                <span
                  className={`font-mono font-bold ${
                    isUngradable ? 'text-rose-700' : 'text-emerald-700'
                  }`}
                >
                  {fundusCase.qualityGroundTruth}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>MobileNetV3 Gate:</span>
                <span className="font-mono text-slate-700">
                  {isUngradable ? 'Rejection' : 'Verified Gradable'}
                </span>
              </div>
              <div className="text-[11px] text-slate-500 pt-1 border-t border-slate-200">
                Note: {fundusCase.notes}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
              <div className="text-[11px] font-mono text-slate-500 font-bold uppercase">
                Preliminary Model Grading
              </div>
              <div className="flex items-center justify-between">
                <span>Tentative Grade:</span>
                <span className="font-mono font-bold text-amber-800">
                  Stage {predictedStage} ({DR_STAGES[predictedStage].shortName})
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>Predictive Entropy:</span>
                <span className="font-mono text-slate-700 font-semibold">{uncertaintyScore.toFixed(3)} bits</span>
              </div>
              <div className="text-[11px] text-slate-500 pt-1 border-t border-slate-200">
                Recommendation:{' '}
                {DR_STAGES[predictedStage].referralRecommended
                  ? 'Dilated fundus slit-lamp exam'
                  : 'Routine 12-month re-screening'}
              </div>
            </div>
          </div>

          {/* Sign-off Block */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
            <div className="text-[11px] font-mono text-slate-600 uppercase font-bold">
              Ophthalmologist Review & Physical Examination
            </div>
            <div className="grid grid-cols-2 gap-6 pt-3 text-slate-600">
              <div className="border-b border-slate-300 pb-8">
                Consultant Ophthalmologist Signature:
              </div>
              <div className="border-b border-slate-300 pb-8">
                Clinical Confirmation Date & Stamp:
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-4 bg-slate-50 border-t border-slate-200">
          <div className="flex items-center gap-2 text-xs text-slate-600 font-mono">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span>RetinaX · Selective Referral Protocol Verified</span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handlePrint}
              className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white font-semibold rounded-lg text-xs font-mono transition-colors flex items-center gap-1.5 shadow-xs"
            >
              <Printer className="w-4 h-4" />
              Print / Save Referral Document
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-800 font-semibold rounded-lg text-xs transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
