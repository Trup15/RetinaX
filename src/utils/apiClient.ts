/**
 * RetinaX API Client
 * Seamlessly bridges the React frontend to the local or remote FastAPI PyTorch backend.
 * Automatically falls back to client-side canvas analysis if backend is unreachable.
 */

const BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

export interface BackendScreeningResponse {
  screening_id: string;
  quality: {
    status: 'GRADABLE' | 'UNGRADABLE';
    gradable_probability: number;
    ungradable_probability: number;
    reason?: string;
    reason_label?: string;
  };
  classification: {
    predicted_stage: 0 | 1 | 2 | 3 | 4;
    calibrated_confidence: number;
    raw_probabilities: number[];
    calibrated_probabilities: number[];
  };
  uncertainty: {
    temperature: number;
    predictive_entropy: number;
    is_uncertain: boolean;
    mc_dropout_variance: number;
    clinical_action: 'ACCEPT_DIAGNOSIS' | 'SPECIALIST_REFERRAL';
  };
  explainability: {
    method: 'GRAD_CAM_PP' | 'SCORE_CAM' | 'INTEGRATED_GRADIENTS';
    heatmap_data_url?: string;
    heatmap_base64?: string;
    lesion_overlap?: {
      dice: number;
      iou: number;
      pointing_game_hit: boolean;
      recall: number;
    };
  };
}

/**
 * Checks if local FastAPI backend server is alive
 */
export async function checkBackendHealth(): Promise<{ online: boolean; device?: string }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1200);

    const res = await fetch(`${BASE_URL}/api/v1/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      return { online: true, device: data.device || 'cpu' };
    }
    return { online: false };
  } catch {
    return { online: false };
  }
}

/**
 * Sends image to Python FastAPI backend
 */
export async function screenFundusImage(
  imageFileOrBlob: Blob,
  eye: 'OD' | 'OS' = 'OD',
  xaiMethod: 'GRAD_CAM_PP' | 'SCORE_CAM' | 'INTEGRATED_GRADIENTS' = 'GRAD_CAM_PP'
): Promise<BackendScreeningResponse> {
  const formData = new FormData();
  formData.append('file', imageFileOrBlob, 'fundus.jpg');
  formData.append('eye', eye);
  formData.append('xai_method', xaiMethod);

  const res = await fetch(`${BASE_URL}/api/v1/screen`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail?.message || `Backend error: HTTP ${res.status}`);
  }

  return res.json();
}
