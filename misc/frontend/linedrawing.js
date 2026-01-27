/**
 * Line Drawing / Sketch Effect Module
 * Converts images into black and white line drawings suitable for tracing
 * 
 * Usage:
 *   const result = LineDrawing.process(imageData, options);
 *   // or
 *   LineDrawing.processToCanvas(sourceCanvas, targetCanvas, options);
 */

const LineDrawing = (function() {
    'use strict';

    // Default options
    const defaultOptions = {
        mode: 'sketch',           // 'sketch', 'outline', 'detailed', 'artistic'
        edgeStrength: 50,         // 0-100: How strong/dark the lines are
        detailLevel: 50,          // 0-100: How much detail to preserve
        lineThickness: 1,         // 1-3: Thickness of detected edges
        invert: false,            // true: white lines on black, false: black lines on white
        smoothing: 1,             // 0-3: Pre-processing blur level
        threshold: 'auto',        // 'auto' or 0-255: Edge detection threshold
        contrastBoost: true       // Enhance contrast before edge detection
    };

    /**
     * Convert RGB to Grayscale using luminosity method
     */
    function rgbToGray(r, g, b) {
        return 0.299 * r + 0.587 * g + 0.114 * b;
    }

    /**
     * Apply Gaussian blur for noise reduction
     */
    function gaussianBlur(pixels, width, height, radius) {
        if (radius === 0) return pixels;
        
        const kernel = generateGaussianKernel(radius);
        const kSize = kernel.length;
        const kHalf = Math.floor(kSize / 2);
        const result = new Float32Array(width * height);
        
        // Horizontal pass
        const temp = new Float32Array(width * height);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let sum = 0;
                let weightSum = 0;
                for (let k = 0; k < kSize; k++) {
                    const px = Math.min(Math.max(x + k - kHalf, 0), width - 1);
                    sum += pixels[y * width + px] * kernel[k];
                    weightSum += kernel[k];
                }
                temp[y * width + x] = sum / weightSum;
            }
        }
        
        // Vertical pass
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let sum = 0;
                let weightSum = 0;
                for (let k = 0; k < kSize; k++) {
                    const py = Math.min(Math.max(y + k - kHalf, 0), height - 1);
                    sum += temp[py * width + x] * kernel[k];
                    weightSum += kernel[k];
                }
                result[y * width + x] = sum / weightSum;
            }
        }
        
        return result;
    }

    /**
     * Generate 1D Gaussian kernel
     */
    function generateGaussianKernel(radius) {
        const size = radius * 2 + 1;
        const kernel = new Float32Array(size);
        const sigma = radius / 2;
        let sum = 0;
        
        for (let i = 0; i < size; i++) {
            const x = i - radius;
            kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
            sum += kernel[i];
        }
        
        // Normalize
        for (let i = 0; i < size; i++) {
            kernel[i] /= sum;
        }
        
        return kernel;
    }

    /**
     * Sobel edge detection
     */
    function sobelEdgeDetection(gray, width, height) {
        const gx = new Float32Array(width * height);
        const gy = new Float32Array(width * height);
        const magnitude = new Float32Array(width * height);
        
        // Sobel kernels
        const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
        
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                let sumX = 0;
                let sumY = 0;
                
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const idx = (y + ky) * width + (x + kx);
                        const kIdx = (ky + 1) * 3 + (kx + 1);
                        sumX += gray[idx] * sobelX[kIdx];
                        sumY += gray[idx] * sobelY[kIdx];
                    }
                }
                
                const i = y * width + x;
                gx[i] = sumX;
                gy[i] = sumY;
                magnitude[i] = Math.sqrt(sumX * sumX + sumY * sumY);
            }
        }
        
        return { gx, gy, magnitude };
    }

    /**
     * Laplacian of Gaussian (LoG) for finer edge detection
     */
    function laplacianOfGaussian(gray, width, height) {
        const result = new Float32Array(width * height);
        
        // 5x5 LoG kernel
        const log = [
            0, 0, 1, 0, 0,
            0, 1, 2, 1, 0,
            1, 2, -16, 2, 1,
            0, 1, 2, 1, 0,
            0, 0, 1, 0, 0
        ];
        
        for (let y = 2; y < height - 2; y++) {
            for (let x = 2; x < width - 2; x++) {
                let sum = 0;
                for (let ky = -2; ky <= 2; ky++) {
                    for (let kx = -2; kx <= 2; kx++) {
                        const idx = (y + ky) * width + (x + kx);
                        const kIdx = (ky + 2) * 5 + (kx + 2);
                        sum += gray[idx] * log[kIdx];
                    }
                }
                result[y * width + x] = Math.abs(sum);
            }
        }
        
        return result;
    }

    /**
     * Non-maximum suppression for thinner edges
     */
    function nonMaxSuppression(magnitude, gx, gy, width, height) {
        const result = new Float32Array(width * height);
        
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const i = y * width + x;
                const mag = magnitude[i];
                
                if (mag === 0) continue;
                
                // Calculate gradient direction
                const angle = Math.atan2(gy[i], gx[i]) * 180 / Math.PI;
                const absAngle = Math.abs(angle);
                
                let neighbor1, neighbor2;
                
                // Determine neighbors based on gradient direction
                if (absAngle < 22.5 || absAngle >= 157.5) {
                    neighbor1 = magnitude[i - 1];
                    neighbor2 = magnitude[i + 1];
                } else if (absAngle >= 22.5 && absAngle < 67.5) {
                    neighbor1 = magnitude[(y - 1) * width + (x + 1)];
                    neighbor2 = magnitude[(y + 1) * width + (x - 1)];
                } else if (absAngle >= 67.5 && absAngle < 112.5) {
                    neighbor1 = magnitude[(y - 1) * width + x];
                    neighbor2 = magnitude[(y + 1) * width + x];
                } else {
                    neighbor1 = magnitude[(y - 1) * width + (x - 1)];
                    neighbor2 = magnitude[(y + 1) * width + (x + 1)];
                }
                
                // Suppress if not local maximum
                if (mag >= neighbor1 && mag >= neighbor2) {
                    result[i] = mag;
                }
            }
        }
        
        return result;
    }

    /**
     * Double threshold and hysteresis for edge linking
     */
    function doubleThreshold(edges, width, height, lowRatio, highRatio) {
        // Find max value
        let maxVal = 0;
        for (let i = 0; i < edges.length; i++) {
            if (edges[i] > maxVal) maxVal = edges[i];
        }
        
        const highThreshold = maxVal * highRatio;
        const lowThreshold = highThreshold * lowRatio;
        
        const result = new Uint8Array(width * height);
        const STRONG = 255;
        const WEAK = 50;
        
        for (let i = 0; i < edges.length; i++) {
            if (edges[i] >= highThreshold) {
                result[i] = STRONG;
            } else if (edges[i] >= lowThreshold) {
                result[i] = WEAK;
            }
        }
        
        // Hysteresis: connect weak edges to strong edges
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const i = y * width + x;
                if (result[i] === WEAK) {
                    // Check 8-connected neighbors for strong edge
                    let hasStrong = false;
                    for (let dy = -1; dy <= 1 && !hasStrong; dy++) {
                        for (let dx = -1; dx <= 1 && !hasStrong; dx++) {
                            if (result[(y + dy) * width + (x + dx)] === STRONG) {
                                hasStrong = true;
                            }
                        }
                    }
                    result[i] = hasStrong ? STRONG : 0;
                }
            }
        }
        
        return result;
    }

    /**
     * Adaptive thresholding for varied lighting conditions
     */
    function adaptiveThreshold(gray, width, height, blockSize, c) {
        const result = new Uint8Array(width * height);
        const halfBlock = Math.floor(blockSize / 2);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let sum = 0;
                let count = 0;
                
                for (let by = -halfBlock; by <= halfBlock; by++) {
                    for (let bx = -halfBlock; bx <= halfBlock; bx++) {
                        const ny = y + by;
                        const nx = x + bx;
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            sum += gray[ny * width + nx];
                            count++;
                        }
                    }
                }
                
                const threshold = (sum / count) - c;
                const i = y * width + x;
                result[i] = gray[i] > threshold ? 255 : 0;
            }
        }
        
        return result;
    }

    /**
     * Pencil sketch effect combining multiple techniques
     */
    function pencilSketchEffect(gray, width, height, options) {
        // Apply different blur amounts based on detail level
        const blurRadius = Math.max(1, Math.floor((100 - options.detailLevel) / 20));
        const blurred = gaussianBlur(gray, width, height, blurRadius);
        
        // Create inverted blurred version
        const invertedBlur = new Float32Array(width * height);
        for (let i = 0; i < gray.length; i++) {
            invertedBlur[i] = 255 - blurred[i];
        }
        
        // Color dodge blend: result = gray / (1 - invertedBlur/255)
        const result = new Float32Array(width * height);
        for (let i = 0; i < gray.length; i++) {
            const base = gray[i];
            const blend = invertedBlur[i];
            
            if (blend >= 255) {
                result[i] = 255;
            } else {
                result[i] = Math.min(255, (base * 255) / (255 - blend));
            }
        }
        
        return result;
    }

    /**
     * Apply contrast enhancement
     */
    function enhanceContrast(gray, width, height) {
        // Find min and max
        let min = 255, max = 0;
        for (let i = 0; i < gray.length; i++) {
            if (gray[i] < min) min = gray[i];
            if (gray[i] > max) max = gray[i];
        }
        
        const range = max - min;
        if (range === 0) return gray;
        
        const result = new Float32Array(width * height);
        for (let i = 0; i < gray.length; i++) {
            result[i] = ((gray[i] - min) / range) * 255;
        }
        
        return result;
    }

    /**
     * Dilate edges for thicker lines
     */
    function dilate(edges, width, height, iterations) {
        let current = edges;
        
        for (let iter = 0; iter < iterations; iter++) {
            const result = new Uint8Array(width * height);
            
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    const i = y * width + x;
                    
                    // Check 3x3 neighborhood
                    let maxVal = 0;
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            const ni = (y + dy) * width + (x + dx);
                            if (current[ni] > maxVal) maxVal = current[ni];
                        }
                    }
                    result[i] = maxVal;
                }
            }
            
            current = result;
        }
        
        return current;
    }

    /**
     * Main processing function
     */
    function process(imageData, options = {}) {
        const opts = { ...defaultOptions, ...options };
        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;
        
        // Convert to grayscale
        const gray = new Float32Array(width * height);
        for (let i = 0; i < width * height; i++) {
            const pi = i * 4;
            gray[i] = rgbToGray(data[pi], data[pi + 1], data[pi + 2]);
        }
        
        // Enhance contrast if enabled
        let processed = opts.contrastBoost ? enhanceContrast(gray, width, height) : gray;
        
        // Apply smoothing
        processed = gaussianBlur(processed, width, height, opts.smoothing);
        
        let edges;
        
        switch (opts.mode) {
            case 'sketch':
                // Pencil sketch effect
                edges = pencilSketchEffect(processed, width, height, opts);
                break;
                
            case 'outline':
                // Strong edge detection with thick lines
                const sobel1 = sobelEdgeDetection(processed, width, height);
                const suppressed1 = nonMaxSuppression(sobel1.magnitude, sobel1.gx, sobel1.gy, width, height);
                edges = doubleThreshold(suppressed1, width, height, 0.3, 0.7);
                if (opts.lineThickness > 1) {
                    edges = dilate(edges, width, height, opts.lineThickness - 1);
                }
                break;
                
            case 'detailed':
                // Combine Sobel and LoG for maximum detail
                const sobel2 = sobelEdgeDetection(processed, width, height);
                const log = laplacianOfGaussian(processed, width, height);
                
                // Combine edge detectors
                const combined = new Float32Array(width * height);
                for (let i = 0; i < combined.length; i++) {
                    combined[i] = Math.max(sobel2.magnitude[i], log[i] * 2);
                }
                
                const suppressed2 = nonMaxSuppression(combined, sobel2.gx, sobel2.gy, width, height);
                const lowThresh = (100 - opts.detailLevel) / 100 * 0.2 + 0.1;
                edges = doubleThreshold(suppressed2, width, height, lowThresh, 0.5);
                break;
                
            case 'artistic':
                // Adaptive threshold for artistic look
                const blockSize = Math.floor(21 + (100 - opts.detailLevel) / 5);
                const c = 5 + (100 - opts.edgeStrength) / 10;
                edges = adaptiveThreshold(processed, width, height, blockSize, c);
                break;
                
            default:
                // Default to sketch
                edges = pencilSketchEffect(processed, width, height, opts);
        }
        
        // Create output ImageData
        const output = new ImageData(width, height);
        const outputData = output.data;
        
        // Apply edge strength and inversion
        const strength = opts.edgeStrength / 100;
        
        for (let i = 0; i < width * height; i++) {
            const pi = i * 4;
            let value;
            
            if (opts.mode === 'sketch') {
                // For sketch mode, darker values are the lines
                value = edges[i];
                // Apply strength by adjusting contrast towards white
                value = 255 - (255 - value) * strength;
            } else {
                // For edge detection modes, white values are the lines
                value = edges[i] * strength;
                value = opts.invert ? value : 255 - value;
            }
            
            value = Math.max(0, Math.min(255, value));
            
            outputData[pi] = value;
            outputData[pi + 1] = value;
            outputData[pi + 2] = value;
            outputData[pi + 3] = 255;
        }
        
        return output;
    }

    /**
     * Process from canvas to canvas
     */
    function processToCanvas(sourceCanvas, targetCanvas, options = {}) {
        const ctx = sourceCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
        
        const result = process(imageData, options);
        
        targetCanvas.width = sourceCanvas.width;
        targetCanvas.height = sourceCanvas.height;
        const targetCtx = targetCanvas.getContext('2d');
        targetCtx.putImageData(result, 0, 0);
        
        return result;
    }

    /**
     * Process from Image element
     */
    function processImage(img, options = {}) {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth || img.width;
        canvas.height = img.naturalHeight || img.height;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        return process(imageData, options);
    }

    /**
     * Get processed image as data URL
     */
    function toDataURL(imageData, format = 'image/png', quality = 0.92) {
        const canvas = document.createElement('canvas');
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        
        const ctx = canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);
        
        return canvas.toDataURL(format, quality);
    }

    /**
     * Download processed image
     */
    function download(imageData, filename = 'line-drawing.png') {
        const dataURL = toDataURL(imageData);
        const link = document.createElement('a');
        link.download = filename;
        link.href = dataURL;
        link.click();
    }

    // Public API
    return {
        process,
        processToCanvas,
        processImage,
        toDataURL,
        download,
        defaultOptions,
        
        // Mode presets
        presets: {
            sketch: { mode: 'sketch', edgeStrength: 60, detailLevel: 50 },
            outline: { mode: 'outline', edgeStrength: 80, detailLevel: 40, lineThickness: 2 },
            detailed: { mode: 'detailed', edgeStrength: 70, detailLevel: 70 },
            artistic: { mode: 'artistic', edgeStrength: 50, detailLevel: 60 },
            coloring: { mode: 'outline', edgeStrength: 90, detailLevel: 30, lineThickness: 2, invert: false },
            trace: { mode: 'detailed', edgeStrength: 85, detailLevel: 60, lineThickness: 1 }
        }
    };
})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LineDrawing;
}
