import React, { useRef, useEffect, useState } from 'react';
import { FundusCase, XAIMethod } from '../types/pipeline';
import { Eye, Crosshair, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

interface RetinalCanvasViewerProps {
  fundusCase: FundusCase;
  xaiMethod: XAIMethod;
  showXAI: boolean;
  xaiOpacity: number;
  saliencyThreshold: number;
  showLesionMasks: boolean;
  showFOVMask: boolean;
  applyCLAHE: boolean;
  activeLesionFilter: 'ALL' | 'MA' | 'HE' | 'EX' | 'SE';
}

export const RetinalCanvasViewer: React.FC<RetinalCanvasViewerProps> = ({
  fundusCase,
  xaiMethod,
  showXAI,
  xaiOpacity,
  saliencyThreshold,
  showLesionMasks,
  showFOVMask,
  applyCLAHE,
  activeLesionFilter,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [hoverCoord, setHoverCoord] = useState<{ x: number; y: number } | null>(null);
  const [cursorActivation, setCursorActivation] = useState<number | null>(null);
  const [cursorLesionType, setCursorLesionType] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = 500;
    const height = 500;
    canvas.width = width;
    canvas.height = height;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = fundusCase.imageUrl;

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Dark background for ophthalmic fundus field
      ctx.fillStyle = '#05070B';
      ctx.fillRect(0, 0, width, height);

      if (img.complete && img.naturalWidth > 0) {
        ctx.drawImage(img, 0, 0, width, height);
      } else {
        // Fallback procedural fundus illustration
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = width * 0.44;

        const grad = ctx.createRadialGradient(centerX, centerY, 20, centerX, centerY, radius);
        grad.addColorStop(0, '#B84518');
        grad.addColorStop(0.7, '#80260B');
        grad.addColorStop(1, '#3B0C04');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fill();

        // Optic Disc
        const discX = centerX - 90;
        const discY = centerY + 10;
        const discGrad = ctx.createRadialGradient(discX, discY, 2, discX, discY, 26);
        discGrad.addColorStop(0, '#FDF3C7');
        discGrad.addColorStop(0.8, '#F59E0B');
        discGrad.addColorStop(1, '#92400E');
        ctx.fillStyle = discGrad;
        ctx.beginPath();
        ctx.arc(discX, discY, 26, 0, Math.PI * 2);
        ctx.fill();

        // Retinal Blood Vessels
        ctx.strokeStyle = '#450A0A';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(discX, discY);
        ctx.bezierCurveTo(discX + 40, discY - 100, centerX + 60, centerY - 120, centerX + 110, centerY - 70);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(discX, discY);
        ctx.bezierCurveTo(discX + 40, discY + 90, centerX + 70, centerY + 120, centerX + 120, centerY + 80);
        ctx.stroke();

        // Macula
        const maculaX = centerX + 50;
        const maculaY = centerY + 10;
        const macGrad = ctx.createRadialGradient(maculaX, maculaY, 2, maculaX, maculaY, 28);
        macGrad.addColorStop(0, '#381008');
        macGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = macGrad;
        ctx.beginPath();
        ctx.arc(maculaX, maculaY, 28, 0, Math.PI * 2);
        ctx.fill();
      }

      // CLAHE Simulation
      if (applyCLAHE) {
        const imgData = ctx.getImageData(0, 0, width, height);
        const data = imgData.data;
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          const enhancedG = Math.min(255, Math.pow(g / 255, 0.75) * 270);
          data[i] = r * 0.9;
          data[i + 1] = enhancedG;
          data[i + 2] = b * 0.85;
        }
        ctx.putImageData(imgData, 0, 0);
      }

      // Circular FOV Mask
      if (showFOVMask) {
        ctx.save();
        ctx.globalCompositeOperation = 'destination-in';
        ctx.beginPath();
        ctx.arc(width / 2, height / 2, width * 0.46, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        ctx.strokeStyle = 'rgba(6, 182, 212, 0.35)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(width / 2, height / 2, width * 0.46, 0, Math.PI * 2);
        ctx.stroke();
      }

      // XAI Heatmap Layer
      if (showXAI && fundusCase.qualityGroundTruth === 'GRADABLE') {
        const heatmapCanvas = document.createElement('canvas');
        heatmapCanvas.width = width;
        heatmapCanvas.height = height;
        const hCtx = heatmapCanvas.getContext('2d');

        if (hCtx) {
          const focalPoints: Array<{ x: number; y: number; r: number; weight: number }> = [];

          if (fundusCase.lesions && fundusCase.lesions.regions.length > 0) {
            fundusCase.lesions.regions.forEach((r) => {
              let weight = 0.9;
              let rad = r.radius * 2.8;
              if (xaiMethod === 'SCORE_CAM') {
                rad = r.radius * 3.8;
                weight = 0.82;
              } else if (xaiMethod === 'INTEGRATED_GRADIENTS') {
                rad = r.radius * 1.6;
                weight = 0.95;
              }
              focalPoints.push({ x: r.x, y: r.y, r: rad, weight });
            });
          } else if (fundusCase.groundTruthStage === 0) {
            focalPoints.push({ x: width * 0.35, y: height * 0.52, r: 40, weight: 0.28 });
          } else {
            focalPoints.push({ x: 260, y: 240, r: 50, weight: 0.85 });
            focalPoints.push({ x: 220, y: 280, r: 40, weight: 0.75 });
          }

          focalPoints.forEach((pt) => {
            const radGrad = hCtx.createRadialGradient(pt.x, pt.y, 2, pt.x, pt.y, pt.r);
            radGrad.addColorStop(0, `rgba(255, 255, 255, ${pt.weight})`);
            radGrad.addColorStop(0.5, `rgba(255, 255, 255, ${pt.weight * 0.5})`);
            radGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
            hCtx.fillStyle = radGrad;
            hCtx.beginPath();
            hCtx.arc(pt.x, pt.y, pt.r, 0, Math.PI * 2);
            hCtx.fill();
          });

          const hData = hCtx.getImageData(0, 0, width, height);
          const px = hData.data;
          const overlayImgData = ctx.getImageData(0, 0, width, height);
          const oPx = overlayImgData.data;

          for (let i = 0; i < px.length; i += 4) {
            const intensity = px[i] / 255;
            if (intensity >= saliencyThreshold) {
              const val = (intensity - saliencyThreshold) / (1 - saliencyThreshold);
              const r = Math.max(0, Math.min(255, Math.floor(255 * (1.5 - Math.abs(val * 4 - 3)))));
              const g = Math.max(0, Math.min(255, Math.floor(255 * (1.5 - Math.abs(val * 4 - 2)))));
              const b = Math.max(0, Math.min(255, Math.floor(255 * (1.5 - Math.abs(val * 4 - 1)))));

              const alpha = xaiOpacity * (0.4 + 0.6 * val);
              oPx[i] = Math.floor(oPx[i] * (1 - alpha) + r * alpha);
              oPx[i + 1] = Math.floor(oPx[i + 1] * (1 - alpha) + g * alpha);
              oPx[i + 2] = Math.floor(oPx[i + 2] * (1 - alpha) + b * alpha);
            }
          }
          ctx.putImageData(overlayImgData, 0, 0);
        }
      }

      // Ground Truth Lesion Masks
      if (showLesionMasks && fundusCase.lesions && fundusCase.lesions.regions.length > 0) {
        fundusCase.lesions.regions.forEach((lesion) => {
          if (activeLesionFilter !== 'ALL' && lesion.type !== activeLesionFilter) {
            return;
          }

          let color = '#F59E0B';
          if (lesion.type === 'MA') {
            color = '#EF4444';
          } else if (lesion.type === 'HE') {
            color = '#DC2626';
          } else if (lesion.type === 'SE') {
            color = '#38BDF8';
          }

          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.fillStyle = `${color}40`;
          ctx.beginPath();
          ctx.arc(lesion.x, lesion.y, lesion.radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();

          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(lesion.x, lesion.y, 1.8, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      // Crosshairs
      if (hoverCoord) {
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.6)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);

        ctx.beginPath();
        ctx.moveTo(0, hoverCoord.y);
        ctx.lineTo(width, hoverCoord.y);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(hoverCoord.x, 0);
        ctx.lineTo(hoverCoord.x, height);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    };

    img.onload = render;
    if (img.complete) {
      render();
    }
  }, [
    fundusCase,
    xaiMethod,
    showXAI,
    xaiOpacity,
    saliencyThreshold,
    showLesionMasks,
    showFOVMask,
    applyCLAHE,
    activeLesionFilter,
    hoverCoord,
  ]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);

    setHoverCoord({ x, y });

    let hitLesion: string | null = null;
    if (fundusCase.lesions && fundusCase.lesions.regions) {
      for (const r of fundusCase.lesions.regions) {
        const dist = Math.hypot(r.x - x, r.y - y);
        if (dist <= r.radius + 3) {
          hitLesion = `${r.type} (${r.type === 'MA' ? 'Microaneurysm' : r.type === 'HE' ? 'Hemorrhage' : r.type === 'EX' ? 'Hard Exudate' : 'Soft Exudate'})`;
          break;
        }
      }
    }
    setCursorLesionType(hitLesion);

    if (fundusCase.lesions && fundusCase.lesions.regions) {
      let maxAct = 0.05;
      for (const r of fundusCase.lesions.regions) {
        const dist = Math.hypot(r.x - x, r.y - y);
        if (dist < 40) {
          const act = Math.max(0, 1 - dist / 40) * 0.95;
          if (act > maxAct) maxAct = act;
        }
      }
      setCursorActivation(Number(maxAct.toFixed(3)));
    } else {
      setCursorActivation(0.04);
    }
  };

  const handleMouseLeave = () => {
    setHoverCoord(null);
    setCursorActivation(null);
    setCursorLesionType(null);
  };

  return (
    <div className="flex flex-col bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
      {/* Viewport Control Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-50 border-b border-slate-200 text-xs">
        <div className="flex items-center gap-3">
          <span className="font-mono text-cyan-700 font-bold flex items-center gap-1.5">
            <Eye className="w-3.5 h-3.5 text-cyan-600" />
            500×500 FOV VIEWPORT
          </span>
          <span className="text-slate-300">|</span>
          <span className="text-slate-600 font-mono">
            {fundusCase.dataset} · {fundusCase.caseNumber} ({fundusCase.eye})
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom((z) => Math.min(2.0, z + 0.25))}
            className="p-1 rounded text-slate-600 hover:text-slate-900 hover:bg-slate-200/70 transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <span className="font-mono text-slate-700 text-[11px] w-10 text-center font-semibold">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom((z) => Math.max(0.75, z - 0.25))}
            className="p-1 rounded text-slate-600 hover:text-slate-900 hover:bg-slate-200/70 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setZoom(1.0)}
            className="p-1 rounded text-slate-600 hover:text-slate-900 hover:bg-slate-200/70 transition-colors"
            title="Reset Zoom"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Canvas Stage */}
      <div
        ref={containerRef}
        className="relative bg-black flex items-center justify-center p-2 overflow-hidden aspect-square select-none cursor-crosshair min-h-[440px]"
      >
        <div
          style={{
            transform: `scale(${zoom})`,
            transformOrigin: 'center center',
            transition: 'transform 0.15s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        >
          <canvas
            ref={canvasRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            className="rounded-lg shadow-inner max-w-full block"
          />
        </div>

        {/* HUD Crosshair Floating Badge */}
        {hoverCoord && (
          <div className="absolute bottom-3 left-3 bg-white/95 backdrop-blur-md border border-slate-200 rounded-md px-2.5 py-1.5 text-[11px] font-mono shadow-md text-slate-800 pointer-events-none flex items-center gap-3">
            <span className="flex items-center gap-1 text-cyan-700 font-bold">
              <Crosshair className="w-3 h-3 text-cyan-600" />
              X: {hoverCoord.x} Y: {hoverCoord.y}
            </span>
            <span>·</span>
            <span>
              Act: <span className="text-amber-700 font-bold">{cursorActivation ?? 0}</span>
            </span>
            {cursorLesionType && (
              <>
                <span>·</span>
                <span className="text-emerald-700 font-bold">{cursorLesionType}</span>
              </>
            )}
          </div>
        )}

        {/* Active Layers Overlay Badge */}
        <div className="absolute top-3 right-3 flex flex-col gap-1 bg-white/95 backdrop-blur-md border border-slate-200 rounded-md p-2 text-[10px] font-mono text-slate-700 shadow-md pointer-events-none">
          <div className="text-slate-900 font-bold mb-0.5">ACTIVE LAYERS</div>
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${showFOVMask ? 'bg-cyan-600' : 'bg-slate-300'}`} />
            <span>FOV Cropping</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${applyCLAHE ? 'bg-emerald-600' : 'bg-slate-300'}`} />
            <span>CLAHE Green Contrast</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${showXAI ? 'bg-amber-500' : 'bg-slate-300'}`} />
            <span>XAI Heatmap (Grad-CAM++)</span>
          </div>
          {fundusCase.hasLesionMasks && (
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${showLesionMasks ? 'bg-rose-500' : 'bg-slate-300'}`} />
              <span>IDRiD Doctor Lesions</span>
            </div>
          )}
        </div>
      </div>

      {/* Colormap Gradient Bar */}
      {showXAI && (
        <div className="px-4 py-2 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-[11px] font-mono text-slate-600">
          <span className="font-semibold text-slate-700">Saliency Activation:</span>
          <div className="flex items-center gap-2 flex-1 max-w-xs mx-4">
            <span className="text-[10px]">0.0</span>
            <div
              className="h-2 flex-1 rounded"
              style={{
                background: 'linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000, #800000)',
              }}
            />
            <span className="text-[10px]">1.0</span>
          </div>
          <span className="font-semibold text-slate-800">Threshold: {saliencyThreshold.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
};
