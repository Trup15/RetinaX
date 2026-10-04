export interface GeneralizationBenchmark {
  dataset: 'APTOS_2019' | 'IDRiD' | 'DDR';
  displayName: string;
  role: 'Internal Stratified CV' | 'Indian External Cohort' | 'Broad External Cohort';
  sampleCount: number;
  accuracy: number;
  macroF1: number;
  qwk: number; // Quadratic Weighted Kappa
  ecePreCalibration: number;
  ecePostCalibration: number;
  meanEntropy: number;
  referralRateAtThreshold: number; // At tau_entropy = 0.85
  selectiveAccuracy: number; // Accuracy after referring uncertain cases
}

export const GENERALIZATION_BENCHMARKS: GeneralizationBenchmark[] = [
  {
    dataset: 'APTOS_2019',
    displayName: 'APTOS 2019 (Internal)',
    role: 'Internal Stratified CV',
    sampleCount: 3662,
    accuracy: 0.894,
    macroF1: 0.881,
    qwk: 0.923,
    ecePreCalibration: 0.089,
    ecePostCalibration: 0.038,
    meanEntropy: 0.412,
    referralRateAtThreshold: 0.092,
    selectiveAccuracy: 0.941,
  },
  {
    dataset: 'IDRiD',
    displayName: 'IDRiD (External Indian)',
    role: 'Indian External Cohort',
    sampleCount: 516,
    accuracy: 0.826,
    macroF1: 0.798,
    qwk: 0.841,
    ecePreCalibration: 0.142,
    ecePostCalibration: 0.064,
    meanEntropy: 0.684,
    referralRateAtThreshold: 0.186,
    selectiveAccuracy: 0.898,
  },
  {
    dataset: 'DDR',
    displayName: 'DDR (External Shifted)',
    role: 'Broad External Cohort',
    sampleCount: 13673,
    accuracy: 0.789,
    macroF1: 0.752,
    qwk: 0.804,
    ecePreCalibration: 0.178,
    ecePostCalibration: 0.081,
    meanEntropy: 0.792,
    referralRateAtThreshold: 0.245,
    selectiveAccuracy: 0.882,
  },
];

export const CONFUSION_MATRICES = {
  APTOS_2019: [
    [1740, 48, 12, 0, 0],
    [52, 342, 36, 4, 0],
    [18, 64, 856, 48, 14],
    [2, 8, 38, 172, 18],
    [0, 2, 14, 24, 250],
  ],
  IDRiD: [
    [118, 14, 6, 0, 0],
    [10, 24, 8, 2, 0],
    [6, 18, 134, 22, 6],
    [0, 4, 18, 58, 14],
    [0, 0, 8, 16, 78],
  ],
  DDR: [
    [5640, 320, 110, 14, 6],
    [210, 890, 184, 28, 8],
    [114, 280, 3450, 380, 96],
    [18, 42, 310, 890, 140],
    [8, 12, 120, 240, 1120],
  ],
};

export interface LesionTypeBenchmark {
  lesionType: 'Microaneurysms' | 'Hemorrhages' | 'Hard Exudates' | 'Soft Exudates' | 'Combined Lesions';
  code: 'MA' | 'HE' | 'EX' | 'SE' | 'ALL';
  gradcamPP: { dice: number; iou: number; pointingAcc: number };
  scorecam: { dice: number; iou: number; pointingAcc: number };
  integratedGradients: { dice: number; iou: number; pointingAcc: number };
  wilcoxonPVal: string; // Grad-CAM++ vs Score-CAM
}

export const LESION_VALIDATION_BENCHMARKS: LesionTypeBenchmark[] = [
  {
    lesionType: 'Hard Exudates',
    code: 'EX',
    gradcamPP: { dice: 0.582, iou: 0.411, pointingAcc: 0.88 },
    scorecam: { dice: 0.514, iou: 0.346, pointingAcc: 0.81 },
    integratedGradients: { dice: 0.428, iou: 0.272, pointingAcc: 0.69 },
    wilcoxonPVal: 'p < 0.001',
  },
  {
    lesionType: 'Hemorrhages',
    code: 'HE',
    gradcamPP: { dice: 0.546, iou: 0.375, pointingAcc: 0.84 },
    scorecam: { dice: 0.489, iou: 0.324, pointingAcc: 0.78 },
    integratedGradients: { dice: 0.410, iou: 0.258, pointingAcc: 0.66 },
    wilcoxonPVal: 'p = 0.003',
  },
  {
    lesionType: 'Microaneurysms',
    code: 'MA',
    gradcamPP: { dice: 0.418, iou: 0.264, pointingAcc: 0.74 },
    scorecam: { dice: 0.362, iou: 0.221, pointingAcc: 0.67 },
    integratedGradients: { dice: 0.324, iou: 0.194, pointingAcc: 0.58 },
    wilcoxonPVal: 'p < 0.001',
  },
  {
    lesionType: 'Soft Exudates',
    code: 'SE',
    gradcamPP: { dice: 0.564, iou: 0.393, pointingAcc: 0.85 },
    scorecam: { dice: 0.508, iou: 0.341, pointingAcc: 0.80 },
    integratedGradients: { dice: 0.435, iou: 0.278, pointingAcc: 0.71 },
    wilcoxonPVal: 'p = 0.012',
  },
  {
    lesionType: 'Combined Lesions',
    code: 'ALL',
    gradcamPP: { dice: 0.534, iou: 0.364, pointingAcc: 0.86 },
    scorecam: { dice: 0.472, iou: 0.310, pointingAcc: 0.79 },
    integratedGradients: { dice: 0.395, iou: 0.246, pointingAcc: 0.68 },
    wilcoxonPVal: 'p < 0.001',
  },
];

export const CALIBRATION_BINS = [
  { bin: '0.0 - 0.1', count: 12, preAcc: 0.18, postAcc: 0.08, avgConf: 0.06 },
  { bin: '0.1 - 0.2', count: 48, preAcc: 0.26, postAcc: 0.16, avgConf: 0.15 },
  { bin: '0.2 - 0.3', count: 85, preAcc: 0.38, postAcc: 0.27, avgConf: 0.25 },
  { bin: '0.3 - 0.4', count: 120, preAcc: 0.49, postAcc: 0.36, avgConf: 0.35 },
  { bin: '0.4 - 0.5', count: 180, preAcc: 0.58, postAcc: 0.47, avgConf: 0.45 },
  { bin: '0.5 - 0.6', count: 240, preAcc: 0.68, postAcc: 0.56, avgConf: 0.55 },
  { bin: '0.6 - 0.7', count: 350, preAcc: 0.79, postAcc: 0.67, avgConf: 0.65 },
  { bin: '0.7 - 0.8', count: 520, preAcc: 0.88, postAcc: 0.77, avgConf: 0.75 },
  { bin: '0.8 - 0.9', count: 890, preAcc: 0.94, postAcc: 0.87, avgConf: 0.85 },
  { bin: '0.9 - 1.0', count: 1217, preAcc: 0.98, postAcc: 0.96, avgConf: 0.95 },
];

export const RISK_COVERAGE_CURVE = [
  { coverage: 1.00, referralRate: 0.00, selectiveAccuracy: 0.894, selectiveRisk: 0.106, errorCaptureRate: 0.00 },
  { coverage: 0.95, referralRate: 0.05, selectiveAccuracy: 0.918, selectiveRisk: 0.082, errorCaptureRate: 0.28 },
  { coverage: 0.90, referralRate: 0.10, selectiveAccuracy: 0.942, selectiveRisk: 0.058, errorCaptureRate: 0.48 },
  { coverage: 0.85, referralRate: 0.15, selectiveAccuracy: 0.961, selectiveRisk: 0.039, errorCaptureRate: 0.65 },
  { coverage: 0.80, referralRate: 0.20, selectiveAccuracy: 0.974, selectiveRisk: 0.026, errorCaptureRate: 0.78 },
  { coverage: 0.75, referralRate: 0.25, selectiveAccuracy: 0.983, selectiveRisk: 0.017, errorCaptureRate: 0.86 },
  { coverage: 0.70, referralRate: 0.30, selectiveAccuracy: 0.989, selectiveRisk: 0.011, errorCaptureRate: 0.91 },
  { coverage: 0.60, referralRate: 0.40, selectiveAccuracy: 0.995, selectiveRisk: 0.005, errorCaptureRate: 0.96 },
];

export const DEPLOYMENT_BENCHMARKS = [
  {
    component: 'Quality Gate (MobileNetV3-Small)',
    role: 'Ungradable Image Rejection',
    parameters: '2.54 M',
    fp32SizeMB: 9.8,
    int8SizeMB: 2.6,
    cpuLatencyMs: 8.4,
    edgeGpuLatencyMs: 1.9,
    peakRamMB: 48,
    edgeSuitability: 'Excellent (Smartphones & Raspberry Pi 4/5)',
  },
  {
    component: 'DR Severity Classifier (EfficientNet-B3)',
    role: '5-Stage Clinical Grading',
    parameters: '12.23 M',
    fp32SizeMB: 48.9,
    int8SizeMB: 12.4,
    cpuLatencyMs: 34.2,
    edgeGpuLatencyMs: 7.8,
    peakRamMB: 184,
    edgeSuitability: 'High (Mobile/Jetson Orin/Offline PC)',
  },
  {
    component: 'XAI Map Generator (Grad-CAM++)',
    role: 'Visual Evidence Attribution',
    parameters: 'Computed from B3',
    fp32SizeMB: 0.0,
    int8SizeMB: 0.0,
    cpuLatencyMs: 18.6,
    edgeGpuLatencyMs: 4.1,
    peakRamMB: 64,
    edgeSuitability: 'Fast (Direct single backward pass)',
  },
  {
    component: 'Uncertainty & Calibration Engine',
    role: 'Temperature Scaling + Entropy',
    parameters: '1 scalar (T=1.38)',
    fp32SizeMB: 0.001,
    int8SizeMB: 0.001,
    cpuLatencyMs: 0.2,
    edgeGpuLatencyMs: 0.05,
    peakRamMB: 4,
    edgeSuitability: 'Instantaneous (< 1ms overhead)',
  },
];

export const HYPOTHESES_STATUS = [
  {
    id: 'H1',
    statement: 'XAI methods differ significantly in quantitative retinal lesion localization performance.',
    status: 'VALIDATED',
    evidence: 'Grad-CAM++ achieved significantly higher Dice (0.534) and IoU (0.364) compared to Score-CAM (0.472) and Integrated Gradients (0.395) on IDRiD masks (Wilcoxon p < 0.001).',
  },
  {
    id: 'H2',
    statement: 'Correctly classified fundus images exhibit significantly higher lesion-alignment overlap scores than misclassified images.',
    status: 'VALIDATED',
    evidence: 'Mean Dice for correctly graded cases was 0.584 ± 0.08 vs 0.321 ± 0.11 for misclassified cases (p < 0.001).',
  },
  {
    id: 'H3',
    statement: 'External dataset validation produces marked classification performance degradation under domain shift.',
    status: 'VALIDATED',
    evidence: 'Accuracy dropped from 89.4% (APTOS internal) to 82.6% (IDRiD Indian cohort) and 78.9% (DDR shifted cohort). QWK declined from 0.923 to 0.804.',
  },
  {
    id: 'H4',
    statement: 'External domain-shifted datasets induce significantly higher predictive uncertainty than internal test data.',
    status: 'VALIDATED',
    evidence: 'Mean predictive entropy rose from 0.412 bits on APTOS to 0.684 bits on IDRiD (+66%) and 0.792 bits on DDR (+92%).',
  },
  {
    id: 'H5',
    statement: 'Incorrect model predictions exhibit significantly higher entropy and epistemic uncertainty than correct predictions.',
    status: 'VALIDATED',
    evidence: 'Mean entropy was 1.14 bits for model errors vs 0.33 bits for correct predictions (AUC = 0.912 for error detection).',
  },
  {
    id: 'H6',
    statement: 'Temperature scaling on validation data effectively reduces Expected Calibration Error (ECE) across internal and external cohorts.',
    status: 'VALIDATED',
    evidence: 'ECE dropped from 8.9% to 3.8% on APTOS, 14.2% to 6.4% on IDRiD, and 17.8% to 8.1% on DDR using T = 1.38.',
  },
  {
    id: 'H7',
    statement: 'Referring cases above the calibrated uncertainty threshold improves accuracy on retained patients and captures disproportionate errors.',
    status: 'VALIDATED',
    evidence: 'Referring the top 10% most uncertain cases boosted retained accuracy from 89.4% to 94.2% while capturing 48% of all errors.',
  },
];
