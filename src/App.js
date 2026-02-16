import { useRef, useEffect } from "react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as THREE from "three";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const startedRef = useRef(false);
  const cubesRef = useRef([]);
  const handStatesRef = useRef({});
  const pinchIndicatorRef = useRef(null);
  const anchorHighlightRef = useRef(null);
  const hoverHighlightRef = useRef(null);
  const rotationHighlightRef = useRef(null);

  const structureGroupRef = useRef(null);

  // Tunable thresholds
  const basePinchStartThreshold = 0.06;
  const basePinchStopThreshold = 0.08;
  const dragThreshold = 0.3;
  const dragAmplification = 1.5;
  const gridUnit = 1;
  const referenceHandSize = 0.2;
  const selectionRadius = 1.0;
  const anchorDwellFrames = 4;
  const generalSmoothingAlpha = 0.3;
  const tipSmoothingAlpha = 0.2;
  const maxVelocity = 0.1;
  const pinchActivationFrames = 3;
  const pinchCooldownMs = 150;
  const predictionFactor = 0.2;

  const moveHoldThresholdMs = 300;
  const movementDeadZone = 0.1;

  // Rotation parameters – improved for natural feel
  const rotationActivationFrames = 3;
  const openHandThreshold = 0.12;          // easier to trigger
  const rotationSmoothingAlpha = 0.1;       // heavy smoothing to remove jitter
  const rotationSpeed = 1.5;                 // natural 1:1 mapping
  const rotationDeadZone = 0.02;             // ignore tiny movements

  const DEBUG = true;

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const structureGroup = new THREE.Group();
    structureGroupRef.current = structureGroup;
    scene.add(structureGroup);

    const camera = new THREE.PerspectiveCamera(75, 640 / 480, 0.1, 1000);
    camera.position.z = 5;
    cameraRef.current = camera;
    const renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setSize(640, 480);
    rendererRef.current = renderer;

    const mainCube = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.8 })
    );
    mainCube.position.set(0, 0, 0);
    mainCube.userData.isOriginal = true;
    structureGroup.add(mainCube);
    cubesRef.current.push(mainCube);

    const outlineGeometry = new THREE.BoxGeometry(1.05, 1.05, 1.05);
    const outlineMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true });
    const outline = new THREE.Mesh(outlineGeometry, outlineMaterial);
    mainCube.add(outline);

    const pinchGeometry = new THREE.SphereGeometry(0.15, 16, 16);
    const pinchMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const pinchIndicator = new THREE.Mesh(pinchGeometry, pinchMaterial);
    pinchIndicator.visible = false;
    scene.add(pinchIndicator);
    pinchIndicatorRef.current = pinchIndicator;

    const anchorHighlightGeometry = new THREE.BoxGeometry(1.1, 1.1, 1.1);
    const anchorHighlightMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00, wireframe: true, transparent: true, opacity: 0.8 });
    const anchorHighlight = new THREE.Mesh(anchorHighlightGeometry, anchorHighlightMaterial);
    anchorHighlight.visible = false;
    scene.add(anchorHighlight);
    anchorHighlightRef.current = anchorHighlight;

    const hoverHighlightGeometry = new THREE.BoxGeometry(1.08, 1.08, 1.08);
    const hoverHighlightMaterial = new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true, transparent: true, opacity: 0.6 });
    const hoverHighlight = new THREE.Mesh(hoverHighlightGeometry, hoverHighlightMaterial);
    hoverHighlight.visible = false;
    scene.add(hoverHighlight);
    hoverHighlightRef.current = hoverHighlight;

    const rotationHighlightGeometry = new THREE.BoxGeometry(1.15, 1.15, 1.15);
    const rotationHighlightMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.3 });
    const rotationHighlight = new THREE.Mesh(rotationHighlightGeometry, rotationHighlightMaterial);
    rotationHighlight.visible = false;
    scene.add(rotationHighlight);
    rotationHighlightRef.current = rotationHighlight;

    const container = document.getElementById("three-container");
    if (container) container.appendChild(renderer.domElement);

    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // ----- Helper functions -----
    const distance3D = (a, b) => {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      return Math.sqrt(dx*dx + dy*dy + dz*dz);
    };

    const initHandState = (handedness) => ({
      handedness,
      smoothed: {
        indexTip: new THREE.Vector3(),
        thumbTip: new THREE.Vector3(),
        wrist: new THREE.Vector3(),
        indexMCP: new THREE.Vector3(),
        pinkyMCP: new THREE.Vector3(),
        middleMCP: new THREE.Vector3(),
      },
      prevSmoothed: null,
      lastUpdateTime: performance.now(),
      pinchState: 'idle',
      pinchFrameCount: 0,
      cooldownUntil: 0,
      isDragging: false,
      interactionMode: null,
      selectedCube: null,
      dragOffset: null,
      pinchStartWorldPos: new THREE.Vector3(),
      pinchStartTime: 0,
      holdTimerActive: false,
      lastSpawnedCube: null,
      spawnAccumX: 0,
      spawnAccumY: 0,
      spawnLastPos: new THREE.Vector3(),
      anchorCandidate: {
        cube: null,
        hitCount: 0,
      },
      framesLost: 0,
      // Rotation state
      rotationState: 'idle',
      rotationFrameCount: 0,
      smoothedYaw: 0,
      prevYaw: 0,
      rotationReference: 0,
      rotationReferenceSet: false,
    });

    const smoothLandmark = (smoothed, raw, alpha) => {
      if (!smoothed) return raw.clone();
      return new THREE.Vector3().lerpVectors(smoothed, raw, alpha);
    };

    const isOutlier = (prev, raw, deltaTime, maxVel) => {
      if (!prev) return false;
      const dist = distance3D(prev, raw);
      const expectedMax = maxVel * (deltaTime / 0.033);
      return dist > expectedMax;
    };

    const normalizedToWorldMirrored = (pos) => {
      const mirroredX = 1 - pos.x;
      return new THREE.Vector3(
        (mirroredX - 0.5) * 10,
        -(pos.y - 0.5) * 10,
        0
      );
    };

    const predictPos = (smoothed, prev) => {
      if (!prev) return smoothed.clone();
      const vel = new THREE.Vector3().subVectors(smoothed, prev);
      return new THREE.Vector3().copy(smoothed).add(vel.multiplyScalar(predictionFactor));
    };

    const isHandOpen = (landmarks) => {
      // Use palm center (wrist + middleMCP) as reference
      const wrist = landmarks[0];
      const middleMCP = landmarks[9];
      const palmCenter = new THREE.Vector3().lerpVectors(wrist, middleMCP, 0.5);
      const fingertips = [4, 8, 12, 16, 20];
      let openCount = 0;
      for (let idx of fingertips) {
        const tip = landmarks[idx];
        const dist = distance3D(palmCenter, tip);
        if (dist > openHandThreshold) openCount++;
      }
      // Require at least 3 fingers open to consider hand open
      return openCount >= 3;
    };

    const isGridOccupied = (pos, excludeCube = null) => {
      const gridPos = new THREE.Vector3(
        Math.round(pos.x / gridUnit) * gridUnit,
        Math.round(pos.y / gridUnit) * gridUnit,
        0
      );
      for (let cube of cubesRef.current) {
        if (excludeCube && cube === excludeCube) continue;
        const cubeGrid = new THREE.Vector3(
          Math.round(cube.position.x / gridUnit) * gridUnit,
          Math.round(cube.position.y / gridUnit) * gridUnit,
          0
        );
        if (cubeGrid.distanceTo(gridPos) < 0.1) return true;
      }
      return false;
    };

    const predictWebcam = async () => {
      if (!canvasRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(videoRef.current, -width, 0, width, height);
      ctx.restore();

      if (handLandmarkerRef.current && videoRef.current) {
        const results = handLandmarkerRef.current.detectForVideo(videoRef.current, Date.now());
        const presentHands = new Set();

        let anyHandDragging = false;
        for (const h in handStatesRef.current) {
          if (handStatesRef.current[h].isDragging) {
            anyHandDragging = true;
            break;
          }
        }

        rotationHighlightRef.current.visible = false;

        if (results.landmarks && results.handedness) {
          results.landmarks.forEach((landmarks, idx) => {
            const handedness = results.handedness[idx][0].categoryName;
            presentHands.add(handedness);

            if (!handStatesRef.current[handedness]) {
              handStatesRef.current[handedness] = initHandState(handedness);
            }
            const handState = handStatesRef.current[handedness];

            // Raw landmarks
            const rawIndexTip = new THREE.Vector3(landmarks[8].x, landmarks[8].y, landmarks[8].z);
            const rawThumbTip = new THREE.Vector3(landmarks[4].x, landmarks[4].y, landmarks[4].z);
            const rawWrist = new THREE.Vector3(landmarks[0].x, landmarks[0].y, landmarks[0].z);
            const rawIndexMCP = new THREE.Vector3(landmarks[5].x, landmarks[5].y, landmarks[5].z);
            const rawPinkyMCP = new THREE.Vector3(landmarks[17].x, landmarks[17].y, landmarks[17].z);
            const rawMiddleMCP = new THREE.Vector3(landmarks[9].x, landmarks[9].y, landmarks[9].z);

            const now = performance.now();
            const deltaTime = now - handState.lastUpdateTime;

            // Outlier rejection
            const prevSmoothed = handState.smoothed;
            let useRawIndex = true, useRawThumb = true, useRawWrist = true;
            let useRawIndexMCP = true, useRawPinkyMCP = true, useRawMiddleMCP = true;
            if (prevSmoothed.indexTip) {
              if (isOutlier(prevSmoothed.indexTip, rawIndexTip, deltaTime, maxVelocity)) useRawIndex = false;
              if (isOutlier(prevSmoothed.thumbTip, rawThumbTip, deltaTime, maxVelocity)) useRawThumb = false;
              if (isOutlier(prevSmoothed.wrist, rawWrist, deltaTime, maxVelocity)) useRawWrist = false;
              if (isOutlier(prevSmoothed.indexMCP, rawIndexMCP, deltaTime, maxVelocity)) useRawIndexMCP = false;
              if (isOutlier(prevSmoothed.pinkyMCP, rawPinkyMCP, deltaTime, maxVelocity)) useRawPinkyMCP = false;
              if (isOutlier(prevSmoothed.middleMCP, rawMiddleMCP, deltaTime, maxVelocity)) useRawMiddleMCP = false;
            }

            // Smooth
            handState.smoothed.indexTip = smoothLandmark(
              useRawIndex ? rawIndexTip : prevSmoothed.indexTip,
              rawIndexTip,
              useRawIndex ? tipSmoothingAlpha : 0
            );
            handState.smoothed.thumbTip = smoothLandmark(
              useRawThumb ? rawThumbTip : prevSmoothed.thumbTip,
              rawThumbTip,
              useRawThumb ? tipSmoothingAlpha : 0
            );
            handState.smoothed.wrist = smoothLandmark(
              useRawWrist ? rawWrist : prevSmoothed.wrist,
              rawWrist,
              useRawWrist ? generalSmoothingAlpha : 0
            );
            handState.smoothed.indexMCP = smoothLandmark(
              useRawIndexMCP ? rawIndexMCP : prevSmoothed.indexMCP,
              rawIndexMCP,
              useRawIndexMCP ? generalSmoothingAlpha : 0
            );
            handState.smoothed.pinkyMCP = smoothLandmark(
              useRawPinkyMCP ? rawPinkyMCP : prevSmoothed.pinkyMCP,
              rawPinkyMCP,
              useRawPinkyMCP ? generalSmoothingAlpha : 0
            );
            handState.smoothed.middleMCP = smoothLandmark(
              useRawMiddleMCP ? rawMiddleMCP : prevSmoothed.middleMCP,
              rawMiddleMCP,
              useRawMiddleMCP ? generalSmoothingAlpha : 0
            );

            handState.prevSmoothed = { ...handState.smoothed };
            handState.lastUpdateTime = now;
            handState.framesLost = 0;

            // Pinch detection
            const handSize = distance3D(handState.smoothed.wrist, handState.smoothed.middleMCP);
            const scaleFactor = handSize / referenceHandSize;
            const pinchStartThreshold = basePinchStartThreshold * scaleFactor;
            const pinchStopThreshold = basePinchStopThreshold * scaleFactor;

            const pinchDist = distance3D(handState.smoothed.indexTip, handState.smoothed.thumbTip);
            const isPinchingRaw = pinchDist < pinchStartThreshold;

            const nowMs = Date.now();
            let pinchActive = false;
            if (handState.pinchState === 'idle') {
              if (isPinchingRaw) {
                handState.pinchState = 'pending';
                handState.pinchFrameCount = 1;
              }
            } else if (handState.pinchState === 'pending') {
              if (isPinchingRaw) {
                handState.pinchFrameCount++;
                if (handState.pinchFrameCount >= pinchActivationFrames) {
                  handState.pinchState = 'active';
                  pinchActive = true;
                }
              } else {
                handState.pinchState = 'idle';
                handState.pinchFrameCount = 0;
              }
            } else if (handState.pinchState === 'active') {
              if (pinchDist < pinchStopThreshold) {
                pinchActive = true;
              } else {
                handState.pinchState = 'cooldown';
                handState.cooldownUntil = nowMs + pinchCooldownMs;
                pinchActive = false;
              }
            } else if (handState.pinchState === 'cooldown') {
              if (nowMs < handState.cooldownUntil) {
                pinchActive = false;
              } else {
                handState.pinchState = 'idle';
              }
            }

            // World positions (mirrored)
            const worldIndexTip = normalizedToWorldMirrored(handState.smoothed.indexTip);
            const worldThumbTip = normalizedToWorldMirrored(handState.smoothed.thumbTip);
            const worldPinchCenter = new THREE.Vector3().lerpVectors(worldIndexTip, worldThumbTip, 0.5);

            // Pinch indicator
            if (pinchActive) {
              pinchIndicatorRef.current.position.copy(worldPinchCenter);
              pinchIndicatorRef.current.visible = true;
            } else {
              pinchIndicatorRef.current.visible = false;
            }

            // ----- Rotation Gesture (Improved) -----
            const worldWrist = normalizedToWorldMirrored(handState.smoothed.wrist);
            const worldMiddleMCP = normalizedToWorldMirrored(handState.smoothed.middleMCP);
            const worldIndexMCP = normalizedToWorldMirrored(handState.smoothed.indexMCP);
            const worldPinkyMCP = normalizedToWorldMirrored(handState.smoothed.pinkyMCP);

            // Palm direction vector (wrist -> middleMCP) – stable and points forward
            const palmDir = new THREE.Vector3().subVectors(worldMiddleMCP, worldWrist).normalize();
            // Yaw angle (rotation around Y axis) – range -PI to PI
            const rawYaw = Math.atan2(palmDir.x, palmDir.z);

            // Smooth yaw (handle wrap-around)
            let deltaYaw = rawYaw - handState.prevYaw;
            if (deltaYaw > Math.PI) deltaYaw -= 2*Math.PI;
            if (deltaYaw < -Math.PI) deltaYaw += 2*Math.PI;
            handState.smoothedYaw = handState.smoothedYaw * rotationSmoothingAlpha + (handState.prevYaw + deltaYaw) * (1 - rotationSmoothingAlpha);
            handState.prevYaw = rawYaw;

            const handIsOpen = isHandOpen(landmarks);
            let rotationActive = false;

            if (!pinchActive) {
              if (handState.rotationState === 'idle') {
                if (handIsOpen) {
                  handState.rotationState = 'pending';
                  handState.rotationFrameCount = 1;
                }
              } else if (handState.rotationState === 'pending') {
                if (handIsOpen) {
                  handState.rotationFrameCount++;
                  if (handState.rotationFrameCount >= rotationActivationFrames) {
                    handState.rotationState = 'active';
                    rotationActive = true;
                    handState.rotationReference = handState.smoothedYaw;
                    handState.rotationReferenceSet = true;
                  }
                } else {
                  handState.rotationState = 'idle';
                  handState.rotationFrameCount = 0;
                }
              } else if (handState.rotationState === 'active') {
                if (handIsOpen) {
                  rotationActive = true;
                } else {
                  handState.rotationState = 'idle';
                  handState.rotationFrameCount = 0;
                }
              }
            } else {
              handState.rotationState = 'idle';
              handState.rotationFrameCount = 0;
            }

            if (rotationActive) {
              // Compute rotation delta from reference
              let yawDelta = handState.smoothedYaw - handState.rotationReference;
              // Normalize to [-PI, PI]
              while (yawDelta > Math.PI) yawDelta -= 2*Math.PI;
              while (yawDelta < -Math.PI) yawDelta += 2*Math.PI;
              if (Math.abs(yawDelta) > rotationDeadZone) {
                structureGroup.rotation.y += yawDelta * rotationSpeed;
                // Update reference to avoid accumulation errors
                handState.rotationReference = handState.smoothedYaw;
              }
              rotationHighlightRef.current.visible = true;
              rotationHighlightRef.current.position.copy(structureGroup.position);
            }

            // ----- Cube selection (only if no other hand is dragging) -----
            if (pinchActive && !handState.isDragging) {
              if (anyHandDragging && !handState.isDragging) {
                // another hand is busy – ignore
              } else {
                let closestCube = null;
                let closestDist = Infinity;
                cubesRef.current.forEach(cube => {
                  const worldCubePos = cube.getWorldPosition(new THREE.Vector3());
                  const dist = worldCubePos.distanceTo(worldIndexTip);
                  if (dist < selectionRadius && dist < closestDist) {
                    closestDist = dist;
                    closestCube = cube;
                  }
                });

                if (closestCube) {
                  if (handState.anchorCandidate.cube === closestCube) {
                    handState.anchorCandidate.hitCount++;
                    if (handState.anchorCandidate.hitCount >= anchorDwellFrames) {
                      handState.selectedCube = closestCube;
                      handState.isDragging = true;
                      handState.interactionMode = null;
                      handState.pinchStartWorldPos.copy(worldPinchCenter);
                      handState.pinchStartTime = performance.now();
                      handState.holdTimerActive = true;

                      anchorHighlightRef.current.position.copy(closestCube.position);
                      anchorHighlightRef.current.visible = true;

                      handState.anchorCandidate.cube = null;
                      handState.anchorCandidate.hitCount = 0;
                    }
                  } else {
                    handState.anchorCandidate.cube = closestCube;
                    handState.anchorCandidate.hitCount = 1;
                  }
                } else {
                  handState.anchorCandidate.cube = null;
                  handState.anchorCandidate.hitCount = 0;
                }
              }
            }

            // ----- Determine interaction mode (spawn vs move) -----
            if (pinchActive && handState.isDragging && handState.interactionMode === null) {
              const elapsed = performance.now() - handState.pinchStartTime;
              const movement = distance3D(worldPinchCenter, handState.pinchStartWorldPos);

              if (movement > movementDeadZone) {
                handState.interactionMode = 'spawn';
                handState.holdTimerActive = false;
                handState.lastSpawnedCube = handState.selectedCube;
                handState.spawnAccumX = 0;
                handState.spawnAccumY = 0;
                handState.spawnLastPos.copy(worldPinchCenter);
              } else if (elapsed > moveHoldThresholdMs) {
                handState.interactionMode = 'move';
                handState.dragOffset = new THREE.Vector3().subVectors(handState.selectedCube.position, worldPinchCenter);
                handState.holdTimerActive = false;
              }
            }

            // ----- Drag logic -----
            if (pinchActive && handState.isDragging && handState.interactionMode !== null) {
              if (handState.interactionMode === 'move') {
                const desiredPos = new THREE.Vector3().copy(worldPinchCenter).add(handState.dragOffset);
                const snappedX = Math.round(desiredPos.x / gridUnit) * gridUnit;
                const snappedY = Math.round(desiredPos.y / gridUnit) * gridUnit;
                const targetPos = new THREE.Vector3(snappedX, snappedY, handState.selectedCube.position.z);

                if (!isGridOccupied(targetPos, handState.selectedCube)) {
                  handState.selectedCube.position.copy(targetPos);
                  anchorHighlightRef.current.position.copy(targetPos);
                }
              } else if (handState.interactionMode === 'spawn') {
                const deltaX = worldPinchCenter.x - handState.spawnLastPos.x;
                const deltaY = worldPinchCenter.y - handState.spawnLastPos.y;
                handState.spawnAccumX += deltaX;
                handState.spawnAccumY += deltaY;
                handState.spawnLastPos.copy(worldPinchCenter);

                while (Math.abs(handState.spawnAccumX) >= gridUnit) {
                  const direction = Math.sign(handState.spawnAccumX);
                  const newPos = handState.lastSpawnedCube.position.clone();
                  newPos.x += direction * gridUnit;
                  newPos.z = handState.lastSpawnedCube.position.z;

                  if (!isGridOccupied(newPos, handState.lastSpawnedCube)) {
                    const newBlock = new THREE.Mesh(
                      new THREE.BoxGeometry(1, 1, 1),
                      new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, opacity: 0.8 })
                    );
                    newBlock.position.copy(newPos);
                    newBlock.userData.isOriginal = false;
                    structureGroup.add(newBlock);
                    cubesRef.current.push(newBlock);
                    const newOutline = new THREE.Mesh(outlineGeometry, outlineMaterial);
                    newBlock.add(newOutline);

                    handState.lastSpawnedCube = newBlock;
                  }
                  handState.spawnAccumX -= direction * gridUnit;
                }

                while (Math.abs(handState.spawnAccumY) >= gridUnit) {
                  const direction = Math.sign(handState.spawnAccumY);
                  const newPos = handState.lastSpawnedCube.position.clone();
                  newPos.y += direction * gridUnit;
                  newPos.z = handState.lastSpawnedCube.position.z;

                  if (!isGridOccupied(newPos, handState.lastSpawnedCube)) {
                    const newBlock = new THREE.Mesh(
                      new THREE.BoxGeometry(1, 1, 1),
                      new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, opacity: 0.8 })
                    );
                    newBlock.position.copy(newPos);
                    newBlock.userData.isOriginal = false;
                    structureGroup.add(newBlock);
                    cubesRef.current.push(newBlock);
                    const newOutline = new THREE.Mesh(outlineGeometry, outlineMaterial);
                    newBlock.add(newOutline);

                    handState.lastSpawnedCube = newBlock;
                  }
                  handState.spawnAccumY -= direction * gridUnit;
                }
              }
            }

            // Release drag
            if (!pinchActive && handState.isDragging) {
              handState.isDragging = false;
              handState.interactionMode = null;
              handState.selectedCube = null;
              handState.dragOffset = null;
              handState.holdTimerActive = false;
              handState.lastSpawnedCube = null;
              handState.spawnAccumX = 0;
              handState.spawnAccumY = 0;
              handState.anchorCandidate.cube = null;
              handState.anchorCandidate.hitCount = 0;
              anchorHighlightRef.current.visible = false;
            }

            // Debug drawing
            if (DEBUG) {
              ctx.fillStyle = "white";
              ctx.font = "16px Arial";
              ctx.fillText(`Pinch dist: ${pinchDist.toFixed(3)}`, 10, 30);
              ctx.fillText(`Mode: ${handState.interactionMode || 'none'}`, 10, 50);
              ctx.fillText(`Hold: ${handState.holdTimerActive ? ((performance.now() - handState.pinchStartTime).toFixed(0)) : '0'}`, 10, 70);
              ctx.fillText(`Accum X: ${handState.spawnAccumX.toFixed(2)}`, 10, 90);
              ctx.fillText(`Accum Y: ${handState.spawnAccumY.toFixed(2)}`, 10, 110);
              ctx.fillText(`Rotation active: ${rotationActive}`, 10, 130);
              ctx.fillText(`Yaw: ${handState.smoothedYaw.toFixed(2)}`, 10, 150);
              ctx.fillText(`Hand open: ${handIsOpen}`, 10, 170);
              ctx.fillText(`Any dragging: ${anyHandDragging}`, 10, 190);

              // Raw landmarks
              ctx.fillStyle = "red";
              ctx.beginPath();
              ctx.arc((1 - rawIndexTip.x) * width, rawIndexTip.y * height, 6, 0, 2 * Math.PI);
              ctx.fill();

              ctx.fillStyle = "orange";
              ctx.beginPath();
              ctx.arc((1 - rawThumbTip.x) * width, rawThumbTip.y * height, 6, 0, 2 * Math.PI);
              ctx.fill();

              // Smoothed
              ctx.fillStyle = "yellow";
              ctx.beginPath();
              ctx.arc((1 - handState.smoothed.indexTip.x) * width, handState.smoothed.indexTip.y * height, 4, 0, 2 * Math.PI);
              ctx.fill();

              ctx.fillStyle = "lime";
              ctx.beginPath();
              ctx.arc((1 - handState.smoothed.thumbTip.x) * width, handState.smoothed.thumbTip.y * height, 4, 0, 2 * Math.PI);
              ctx.fill();

              // Midpoint
              ctx.fillStyle = "cyan";
              ctx.beginPath();
              ctx.arc((1 - (handState.smoothed.indexTip.x + handState.smoothed.thumbTip.x)/2) * width,
                      ((handState.smoothed.indexTip.y + handState.smoothed.thumbTip.y)/2) * height,
                      3, 0, 2 * Math.PI);
              ctx.fill();
            }
          });

          // Global hover highlight
          let hoverCube = null;
          let minDist = Infinity;
          for (const handedness in handStatesRef.current) {
            const handState = handStatesRef.current[handedness];
            if (!handState.isDragging && handState.smoothed.indexTip) {
              const worldTip = normalizedToWorldMirrored(handState.smoothed.indexTip);
              cubesRef.current.forEach(cube => {
                const worldCubePos = cube.getWorldPosition(new THREE.Vector3());
                const dist = worldCubePos.distanceTo(worldTip);
                if (dist < selectionRadius && dist < minDist) {
                  minDist = dist;
                  hoverCube = cube;
                }
              });
            }
          }
          if (hoverCube) {
            hoverHighlightRef.current.position.copy(hoverCube.position);
            hoverHighlightRef.current.visible = true;
          } else {
            hoverHighlightRef.current.visible = false;
          }
        }

        // Handle lost hands
        for (const handedness in handStatesRef.current) {
          if (!presentHands.has(handedness)) {
            const handState = handStatesRef.current[handedness];
            handState.framesLost++;
            if (handState.framesLost > 30 && handState.isDragging) {
              handState.isDragging = false;
              handState.interactionMode = null;
              handState.selectedCube = null;
              handState.dragOffset = null;
              handState.holdTimerActive = false;
              handState.lastSpawnedCube = null;
              handState.spawnAccumX = 0;
              handState.spawnAccumY = 0;
              handState.anchorCandidate.cube = null;
              handState.anchorCandidate.hitCount = 0;
              anchorHighlightRef.current.visible = false;
              handState.pinchState = 'idle';
              handState.rotationState = 'idle';
            }
          }
        }
      }

      requestAnimationFrame(predictWebcam);
    };

    const runHandLandmarker = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      );
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      if (videoRef.current) {
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then((stream) => {
            videoRef.current.srcObject = stream;
            videoRef.current.addEventListener("loadeddata", predictWebcam);
          })
          .catch((err) => console.error("Camera error:", err));
      }
    };

    runHandLandmarker();
  }, []);

  return (
    <div style={{ textAlign: "center", position: "relative" }}>
      <h2>Natural Palm Rotation – Just Open Your Hand</h2>
      <div id="three-container" style={{ position: "absolute", top: 50, left: "50%", transform: "translateX(-50%)" }}></div>
      <video ref={videoRef} autoPlay playsInline width="640" height="480" style={{ display: "none" }} />
      <canvas ref={canvasRef} width="640" height="480" />
      <p>
        <strong>Rotation:</strong> Open your hand (fingers spread) and rotate your palm – the structure follows smoothly.<br/>
        No jitter, no over‑sensitivity – just natural control. Debug overlay shows when rotation is active.
      </p>
    </div>
  );
}

export default App;