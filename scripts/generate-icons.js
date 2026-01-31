#!/usr/bin/env node

/**
 * Generate app icons based on the Echo design from Pencil
 * Design: Sage green background with concentric white wave circles
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Design specs from echo.pen
const DESIGN = {
  background: '#7C9082',
  shadow: { color: '#7C908250', blur: 40, offsetY: 8 },
  waves: [
    { opacity: 0.19, thickness: 3, size: 0.72 },  // wave1: 144/200
    { opacity: 0.31, thickness: 4, size: 0.56 },  // wave2: 112/200
    { opacity: 0.44, thickness: 5, size: 0.40 },  // wave3: 80/200
  ],
  core: { size: 0.24, glow: { blur: 20, spread: 4, opacity: 0.375 } }
};

// Required icon sizes for Tauri macOS app
const ICON_SIZES = [
  { name: '32x32.png', size: 32 },
  { name: '64x64.png', size: 64 },
  { name: '128x128.png', size: 128 },
  { name: '128x128@2x.png', size: 256 },
  { name: 'icon.png', size: 1024 },
];

function generateSVG(size) {
  const cornerRadius = size * 0.22;
  const center = size / 2;

  // Calculate wave dimensions
  const waves = DESIGN.waves.map((w, i) => {
    const waveSize = size * w.size;
    const strokeWidth = (w.thickness / 200) * size;
    const opacity = w.opacity;
    return { size: waveSize, strokeWidth, opacity };
  });

  const coreSize = size * DESIGN.core.size;
  const glowBlur = (DESIGN.core.glow.blur / 200) * size;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Drop shadow for background -->
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="${(8/200)*size}" stdDeviation="${(20/200)*size}" flood-color="#7C9082" flood-opacity="0.31"/>
    </filter>
    <!-- Glow for core -->
    <filter id="glow" x="-100%" y="-100%" width="300%" height="300%">
      <feGaussianBlur stdDeviation="${glowBlur/2}" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- Background -->
  <rect x="0" y="0" width="${size}" height="${size}" rx="${cornerRadius}" ry="${cornerRadius}" fill="${DESIGN.background}" filter="url(#shadow)"/>

  <!-- Wave 1 (outermost) -->
  <ellipse cx="${center}" cy="${center}" rx="${waves[0].size/2}" ry="${waves[0].size/2}"
           fill="none" stroke="white" stroke-opacity="${waves[0].opacity}" stroke-width="${waves[0].strokeWidth}"/>

  <!-- Wave 2 -->
  <ellipse cx="${center}" cy="${center}" rx="${waves[1].size/2}" ry="${waves[1].size/2}"
           fill="none" stroke="white" stroke-opacity="${waves[1].opacity}" stroke-width="${waves[1].strokeWidth}"/>

  <!-- Wave 3 (innermost) -->
  <ellipse cx="${center}" cy="${center}" rx="${waves[2].size/2}" ry="${waves[2].size/2}"
           fill="none" stroke="white" stroke-opacity="${waves[2].opacity}" stroke-width="${waves[2].strokeWidth}"/>

  <!-- Core with glow -->
  <ellipse cx="${center}" cy="${center}" rx="${coreSize/2}" ry="${coreSize/2}"
           fill="white" filter="url(#glow)"/>
</svg>`;
}

function main() {
  const iconsDir = path.join(__dirname, '..', 'src-tauri', 'icons');
  const tempDir = path.join(__dirname, '..', '.icon-temp');

  // Create temp directory
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  console.log('Generating Echo app icons...\n');

  // Generate SVG and PNG for each size
  for (const icon of ICON_SIZES) {
    const svg = generateSVG(icon.size);
    const svgPath = path.join(tempDir, `icon-${icon.size}.svg`);
    const pngPath = path.join(iconsDir, icon.name);

    // Write SVG
    fs.writeFileSync(svgPath, svg);

    // Convert to PNG using rsvg-convert or sips (macOS)
    try {
      // Try rsvg-convert first (higher quality)
      execSync(`rsvg-convert -w ${icon.size} -h ${icon.size} "${svgPath}" -o "${pngPath}"`, { stdio: 'pipe' });
      console.log(`✓ Generated ${icon.name} (${icon.size}x${icon.size})`);
    } catch {
      // Fallback to sips on macOS
      try {
        // sips can't convert SVG directly, so we'll keep the SVG for now
        console.log(`⚠ Skipping ${icon.name} - rsvg-convert not found`);
      } catch (e) {
        console.error(`✗ Failed to generate ${icon.name}:`, e.message);
      }
    }
  }

  // Generate icns for macOS using iconutil
  console.log('\nGenerating icon.icns...');
  const iconsetDir = path.join(tempDir, 'icon.iconset');
  if (!fs.existsSync(iconsetDir)) {
    fs.mkdirSync(iconsetDir, { recursive: true });
  }

  // macOS iconset requires specific sizes
  const iconsetSizes = [
    { name: 'icon_16x16.png', size: 16 },
    { name: 'icon_16x16@2x.png', size: 32 },
    { name: 'icon_32x32.png', size: 32 },
    { name: 'icon_32x32@2x.png', size: 64 },
    { name: 'icon_128x128.png', size: 128 },
    { name: 'icon_128x128@2x.png', size: 256 },
    { name: 'icon_256x256.png', size: 256 },
    { name: 'icon_256x256@2x.png', size: 512 },
    { name: 'icon_512x512.png', size: 512 },
    { name: 'icon_512x512@2x.png', size: 1024 },
  ];

  let icnsSuccess = true;
  for (const icon of iconsetSizes) {
    const svg = generateSVG(icon.size);
    const svgPath = path.join(tempDir, `iconset-${icon.size}.svg`);
    const pngPath = path.join(iconsetDir, icon.name);

    fs.writeFileSync(svgPath, svg);

    try {
      execSync(`rsvg-convert -w ${icon.size} -h ${icon.size} "${svgPath}" -o "${pngPath}"`, { stdio: 'pipe' });
    } catch {
      icnsSuccess = false;
    }
  }

  if (icnsSuccess) {
    try {
      execSync(`iconutil -c icns "${iconsetDir}" -o "${path.join(iconsDir, 'icon.icns')}"`, { stdio: 'pipe' });
      console.log('✓ Generated icon.icns');
    } catch (e) {
      console.log('⚠ Failed to generate icon.icns:', e.message);
    }
  }

  // Cleanup temp directory
  fs.rmSync(tempDir, { recursive: true, force: true });

  console.log('\nDone! Icon files have been updated in src-tauri/icons/');
}

main();
