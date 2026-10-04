import React, { useState } from 'react';
import { LivePipelineConsole } from './components/LivePipelineConsole';
import { LesionValidationLab } from './components/LesionValidationLab';
import { CrossDatasetShiftExplorer } from './components/CrossDatasetShiftExplorer';
import { UncertaintyReferralLab } from './components/UncertaintyReferralLab';
import { OfflineEdgeBenchmark } from './components/OfflineEdgeBenchmark';
import { ResearchBlueprintViewer } from './components/ResearchBlueprintViewer';
import { ReferralModal } from './components/ReferralModal';
import { FundusCase, DRStage } from './types/pipeline';
import { CLINICAL_CASES } from './data/sampleCohorts';
import { FileText, Download } from 'lucide-react';

type NavigationTab =
  | 'pipeline'
  | 'lesions'
  | 'generalization'
  | 'uncertainty'
  | 'deployment'
  | 'blueprint';

export default function App() {
  const [activeTab, setActiveTab] = useState<NavigationTab>('pipeline');
  const [referralModalCase, setReferralModalCase] = useState<{
    fundusCase: FundusCase;
    uncertaintyScore: number;
    predictedStage: DRStage;
  } | null>(null);

  const handleOpenReferralModal = (
    fundusCase: FundusCase,
    uncertaintyScore: number,
    predictedStage: DRStage
  ) => {
    setReferralModalCase({ fundusCase, uncertaintyScore, predictedStage });
  };

  const handleExportSummaryReport = () => {
    // Generate a printable summary of the screening session
    window.print();
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-200">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="sticky top-0 z-40 bg-slate-950/90 backdrop-blur-md border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Zone 1: Single text element wordmark */}
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('pipeline');
            }}
            className="text-lg font-bold tracking-tight text-slate-100 hover:text-cyan-400 transition-colors whitespace-nowrap"
          >
            RetinaTrust AI
          </a>

          {/* Zone 2: 4-6 clean text navigation links */}
          <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-slate-400">
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`hover:text-slate-100 transition-colors whitespace-nowrap pb-0.5 ${
                activeTab === 'pipeline'
                  ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
                  : ''
              }`}
            >
              Live Pipeline
            </button>
            <button
              onClick={() => setActiveTab('lesions')}
              className={`hover:text-slate-100 transition-colors whitespace-nowrap pb-0.5 ${
                activeTab === 'lesions'
                  ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
                  : ''
              }`}
            >
              Lesion XAI
            </button>
            <button
              onClick={() => setActiveTab('generalization')}
              className={`hover:text-slate-100 transition-colors whitespace-nowrap pb-0.5 ${
                activeTab === 'generalization'
                  ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
                  : ''
              }`}
            >
              Cross-Dataset
            </button>
            <button
              onClick={() => setActiveTab('uncertainty')}
              className={`hover:text-slate-100 transition-colors whitespace-nowrap pb-0.5 ${
                activeTab === 'uncertainty'
                  ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
                  : ''
              }`}
            >
              Uncertainty
            </button>
            <button
              onClick={() => setActiveTab('deployment')}
              className={`hover:text-slate-100 transition-colors whitespace-nowrap pb-0.5 ${
                activeTab === 'deployment'
                  ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
                  : ''
              }`}
            >
              Edge Benchmark
            </button>
            <button
              onClick={() => setActiveTab('blueprint')}
              className={`hover:text-slate-100 transition-colors whitespace-nowrap pb-0.5 ${
                activeTab === 'blueprint'
                  ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
                  : ''
              }`}
            >
              Blueprint
            </button>
          </nav>

          {/* Zone 3: 1-2 primary actions */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleExportSummaryReport}
              className="px-3.5 py-1.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors whitespace-nowrap flex items-center gap-1.5 font-mono shadow-sm shadow-cyan-500/20"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Report</span>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Row (Horizontal Scrollable) */}
        <div className="md:hidden flex items-center gap-4 px-4 py-2 border-t border-slate-800/80 overflow-x-auto text-xs font-medium text-slate-400">
          {(
            [
              { id: 'pipeline', label: 'Live Pipeline' },
              { id: 'lesions', label: 'Lesion XAI' },
              { id: 'generalization', label: 'Cross-Dataset' },
              { id: 'uncertainty', label: 'Uncertainty' },
              { id: 'deployment', label: 'Edge Benchmark' },
              { id: 'blueprint', label: 'Blueprint' },
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`whitespace-nowrap px-2 py-1 rounded transition-colors ${
                activeTab === tab.id
                  ? 'text-cyan-300 bg-slate-900 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </header>

      {/* Main Container Stage */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'pipeline' && (
          <LivePipelineConsole onOpenReferralModal={handleOpenReferralModal} />
        )}
        {activeTab === 'lesions' && <LesionValidationLab />}
        {activeTab === 'generalization' && <CrossDatasetShiftExplorer />}
        {activeTab === 'uncertainty' && <UncertaintyReferralLab />}
        {activeTab === 'deployment' && <OfflineEdgeBenchmark />}
        {activeTab === 'blueprint' && <ResearchBlueprintViewer />}
      </main>

      {/* Specialist Referral Modal */}
      {referralModalCase && (
        <ReferralModal
          fundusCase={referralModalCase.fundusCase}
          uncertaintyScore={referralModalCase.uncertaintyScore}
          predictedStage={referralModalCase.predictedStage}
          onClose={() => setReferralModalCase(null)}
        />
      )}

      {/* Quiet Academic / Clinical Footer */}
      <footer className="mt-auto border-t border-slate-900 bg-slate-950/60 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span>RetinaTrust AI Framework</span>
            <span aria-hidden="true">·</span>
            <span>Indian Retinal Fundus Screening Research</span>
            <span aria-hidden="true">·</span>
            <span>EfficientNet-B3 + MobileNetV3-Small</span>
          </div>
          <div className="flex items-center gap-3 font-mono text-[11px] text-slate-600">
            <span>IDRiD Lesion GT</span>
            <span aria-hidden="true">·</span>
            <span>APTOS 2019</span>
            <span aria-hidden="true">·</span>
            <span>DDR Cohort</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
