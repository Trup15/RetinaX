export type DRStage = 0 | 1 | 2 | 3 | 4;

export interface DRStageInfo {
  stage: DRStage;
  name: string;
  shortName: string;
  description: string;
  clinicalSigns: string[];
  referralRecommended: boolean;
  color: string;
}

export const DR_STAGES: Record<DRStage, DRStageInfo> = {
  0: {
    stage: 0,
    name: 'No Apparent Retinopathy',
    shortName: 'No DR',
    description: 'No retinal vascular abnormalities detected. Normal macula and optic disc architecture.',
    clinicalSigns: ['Clear retina', 'Uniform vasculature', 'Crisp disc margins', 'No microaneurysms'],
    referralRecommended: false,
    color: '#10B981', // emerald
  },
  1: {
    stage: 1,
    name: 'Mild Non-Proliferative DR',
    shortName: 'Mild NPDR',
    description: 'Presence of isolated microaneurysms only. Earliest clinically observable stage.',
    clinicalSigns: ['Isolated microaneurysms', 'Focal capillary out-pouching', 'No macular edema'],
    referralRecommended: false,
    color: '#06B6D4', // cyan
  },
  2: {
    stage: 2,
    name: 'Moderate Non-Proliferative DR',
    shortName: 'Moderate NPDR',
    description: 'Microaneurysms accompanied by retinal blot hemorrhages, hard exudates, or early cotton wool spots.',
    clinicalSigns: ['Multiple microaneurysms', 'Blot hemorrhages', 'Hard waxy exudates', 'Occasional soft exudate'],
    referralRecommended: true,
    color: '#F59E0B', // amber
  },
  3: {
    stage: 3,
    name: 'Severe Non-Proliferative DR',
    shortName: 'Severe NPDR',
    description: 'Meets 4-2-1 rule: diffuse intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant.',
    clinicalSigns: ['4-2-1 rule met', 'Venous beading', 'Intraretinal microvascular abnormalities (IRMA)', 'Substantial ischemic threat'],
    referralRecommended: true,
    color: '#F97316', // orange
  },
  4: {
    stage: 4,
    name: 'Proliferative Diabetic Retinopathy',
    shortName: 'PDR',
    description: 'Hallmarked by pathological neovascularization (NVD/NVE), pre-retinal/vitreous hemorrhages, and fibrovascular proliferation.',
    clinicalSigns: ['Neovascularization at disc (NVD)', 'Neovascularization elsewhere (NVE)', 'Vitreous/preretinal hemorrhage', 'High vision loss risk'],
    referralRecommended: true,
    color: '#EF4444', // red
  },
};

export type QualityGrade = 'GRADABLE' | 'UNGRADABLE';

export type UngradableReason =
  | 'SEVERE_OPTICAL_BLUR'
  | 'MEDIA_OPACITY_CATARACT'
  | 'ILLUMINATION_NON_UNIFORMITY'
  | 'PERIPHERAL_FIELD_CLIPPING';

export interface QualityAssessment {
  status: QualityGrade;
  gradableProbability: number;
  ungradableProbability: number;
  reason?: UngradableReason;
  reasonLabel?: string;
  recommendedAction: 'PROCEED_TO_CLASSIFIER' | 'RECAPTURE_IMAGE' | 'REFER_FOR_DILATED_EXAM';
}

export type XAIMethod = 'GRAD_CAM_PP' | 'SCORE_CAM' | 'INTEGRATED_GRADIENTS';

export interface LesionTypeMask {
  id: 'MA' | 'HE' | 'EX' | 'SE';
  name: string;
  count: number;
  areaPixels: number;
  color: string;
  description: string;
}

export interface LesionValidationMetrics {
  iou: number; // Intersection over Union
  dice: number; // Dice Similarity Coefficient
  pointingGameHit: boolean; // Peak activation inside lesion
  lesionRecall: number; // Percentage of lesion area covered
  lesionPrecision: number;
  meanActivationInsideLesion: number;
  meanActivationOutsideLesion: number;
}

export interface MCDropoutMetrics {
  sampleCount: number;
  classMeanProbabilities: number[];
  classStdDeviations: number[];
  epistemicUncertainty: number; // Variance across passes
}

export interface UncertaintyEvaluation {
  maxSoftmaxProb: number;
  rawSoftmaxProb: number[];
  temperature: number;
  calibratedProb: number[];
  calibratedConfidence: number;
  predictiveEntropy: number; // In bits
  mcDropout: MCDropoutMetrics;
  isUncertain: boolean;
  uncertaintyScore: number; // Normalized 0-1
  referralDecision: 'ACCEPT_DIAGNOSIS' | 'SPECIALIST_REFERRAL';
  referralRationale: string;
}

export interface FundusCase {
  id: string;
  caseNumber: string;
  dataset: 'APTOS_2019' | 'IDRiD' | 'DDR' | 'USER_UPLOAD';
  patientAge: number;
  patientGender: 'M' | 'F';
  eye: 'OD' | 'OS'; // Right or Left
  imageUrl: string;
  groundTruthStage: DRStage;
  qualityGroundTruth: QualityGrade;
  hasLesionMasks: boolean;
  lesions?: {
    microaneurysms: number;
    hemorrhages: number;
    hardExudates: number;
    softExudates: number;
    maskDataUrl?: string;
    regions: Array<{
      type: 'MA' | 'HE' | 'EX' | 'SE';
      x: number;
      y: number;
      radius: number;
    }>;
  };
  notes: string;
}
