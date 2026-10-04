import React, { useState } from 'react';
import { SimpleScreener } from './components/SimpleScreener';
import { SimpleLesionValidation } from './components/SimpleLesionValidation';
import { SimpleResearchSummary } from './components/SimpleResearchSummary';
import { ReferralModal } from './components/ReferralModal';
import { FundusCase, DRStage } from './types/pipeline';
import { Printer } from 'lucide-react';

type Tab = 'screener' | 'validation' | 'safety';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('screener');
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

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-900">
      {/* 3-Zone Clean Header */}
      <header className="sticky top-0 z-40 bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Zone 1: Single text element wordmark */}
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('screener');
            }}
            className="text-lg font-bold tracking-tight text-slate-900 hover:text-cyan-600 transition-colors whitespace-nowrap"
          >
            RetinaX
          </a>

          {/* Zone 2: 3 clear navigation links */}
          <nav className="flex items-center gap-6 text-xs font-semibold text-slate-600">
            <button
              onClick={() => setActiveTab('screener')}
              className={`hover:text-slate-900 transition-colors whitespace-nowrap pb-0.5 ${
                activeTab === 'screener'
                  ? 'text-cyan-600 border-b-2 border-cyan-600 font-bold'
                  : ''
              }`}
            >
              Retinal Screener
            </button>
            <button
              onClick={() => setActiveTab('validation')}
              className={`hover:text-slate-900 transition-colors whitespace-nowrap pb-0.5 ${
                activeTab === 'validation'
                  ? 'text-cyan-600 border-b-2 border-cyan-600 font-bold'
                  : ''
              }`}
            >
              Lesion AI Check
            </button>
            <button
              onClick={() => setActiveTab('safety')}
              className={`hover:text-slate-900 transition-colors whitespace-nowrap pb-0.5 ${
                activeTab === 'safety'
                  ? 'text-cyan-600 border-b-2 border-cyan-600 font-bold'
                  : ''
              }`}
            >
              Safety & Real-World Tests
            </button>
          </nav>

          {/* Zone 3: 1 primary action */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => window.print()}
              className="px-3.5 py-1.5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors whitespace-nowrap flex items-center gap-1.5 shadow-sm"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print Page</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'screener' && (
          <SimpleScreener onOpenReferralModal={handleOpenReferralModal} />
        )}
        {activeTab === 'validation' && <SimpleLesionValidation />}
        {activeTab === 'safety' && <SimpleResearchSummary />}
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

      {/* Clean Uncluttered Footer */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-5 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-700">RetinaX Framework</span>
            <span aria-hidden="true">·</span>
            <span>Indian Retinal Fundus Screening</span>
            <span aria-hidden="true">·</span>
            <span>Image Quality + 5-Stage Grading + Lesion Validation + Safe Referral</span>
          </div>
          <div className="text-slate-500 font-mono text-[11px]">
            IDRiD · APTOS 2019 · DDR
          </div>
        </div>
      </footer>
    </div>
  );
}

