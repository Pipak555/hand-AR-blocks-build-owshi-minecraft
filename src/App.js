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
  const cubesRef = useRef([]);  // Array of all cubes (for raycasting against any)
  const handStatesRef = useRef([]);  // Per-hand state: { isDragging, anchorCube, lastSpawnPos, pinchStartPos }
  const pinchIndicatorRef = useRef(null);  // Visual indicator for pinch
  const highlightRef = useRef(null);  // Optional highlight for anchor cube

  // Tunable thresholds
  const pinchThreshold = 0.05;
  const dragThreshold = 0.5;  // Distance to drag before checking snap
  const gridUnit = 1;  // Grid size (1 unit = full cube size)

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    // Setup Three.js scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(75, 640 / 480, 0.1, 1000);
    camera.position.z = 5;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setSize(640, 480);
    rendererRef.current = renderer;

    // Main anchor cube (fixed in center)
    const mainCube = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.8 })
    );
    mainCube.position.set(0, 0, 0);
    scene.add(mainCube);
    cubesRef.current.push(mainCube);

    // Add wireframe outline
    const outlineGeometry = new THREE.BoxGeometry(1.05, 1.05, 1.05);
    const outlineMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true });
    const outline = new THREE.Mesh(outlineGeometry, outlineMaterial);
    mainCube.add(outline);

    // Pinch indicator (red sphere)
    const pinchGeometry = new THREE.SphereGeometry(0.1, 16, 16);
    const pinchMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
    const pinchIndicator = new THREE.Mesh(pinchGeometry, pinchMaterial);
    pinchIndicator.visible = false;
    scene.add(pinchIndicator);
    pinchIndicatorRef.current = pinchIndicator;

    // Optional highlight for anchor cube (yellow outline)
    const highlightGeometry = new THREE.BoxGeometry(1.1, 1.1, 1.1);
    const highlightMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00, wireframe: true, transparent: true, opacity: 0.5 });
    const highlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
    highlight.visible = false;
    scene.add(highlight);
    highlightRef.current = highlight;

    // Append renderer
    const container = document.getElementById("three-container");
    if (container) container.appendChild(renderer.domElement);

    // Animate Three.js
    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // Hand tracking setup (numHands: 2)
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

    const predictWebcam = async () => {
      if (!canvasRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      // Camera inversion
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(videoRef.current, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      if (handLandmarkerRef.current && videoRef.current) {
        const results = handLandmarkerRef.current.detectForVideo(videoRef.current, Date.now());

        if (results.landmarks) {
          results.landmarks.forEach((landmarks, index) => {
            // Initialize per-hand state if not exists
            if (!handStatesRef.current[index]) {
              handStatesRef.current[index] = {
                isDragging: false,
                anchorCube: null,
                lastSpawnPos: null,
                pinchStartPos: null
              };
            }
            const handState = handStatesRef.current[index];

            console.log(`Hand ${index + 1} landmarks:`, landmarks.length);
            const indexTip = landmarks[8];
            const thumbTip = landmarks[4];
            console.log(`Hand ${index + 1} index tip: (${indexTip.x.toFixed(2)}, ${indexTip.y.toFixed(2)}), thumb tip: (${thumbTip.x.toFixed(2)}, ${thumbTip.y.toFixed(2)})`);

            const color = index === 0 ? "red" : "blue";
            ctx.fillStyle = color;
            ctx.strokeStyle = color;

            for (const landmark of landmarks) {
              ctx.beginPath();
              ctx.arc((1 - landmark.x) * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
              ctx.fill();
            }

            const wrist = landmarks[0];
            ctx.fillText(`Hand ${index + 1}`, (1 - wrist.x) * canvas.width, wrist.y * canvas.height - 20);

            // Mirrored key points
            const indexTipMirrored = { x: 1 - landmarks[8].x, y: landmarks[8].y };
            const thumbTipMirrored = { x: 1 - landmarks[4].x, y: landmarks[4].y };

            // Pinch detection
            const pinchDistance = Math.sqrt(
              Math.pow(indexTipMirrored.x - thumbTipMirrored.x, 2) + Math.pow(indexTipMirrored.y - thumbTipMirrored.y, 2)
            );
            const isPinching = pinchDistance < pinchThreshold;
            console.log(`Hand ${index + 1} pinch distance: ${pinchDistance.toFixed(2)}, pinching: ${isPinching}`);

            // Show pinch indicator
            if (isPinching) {
              pinchIndicatorRef.current.position.set(
                (indexTipMirrored.x - 0.5) * 10,
                -(indexTipMirrored.y - 0.5) * 10,
                0
              );
              pinchIndicatorRef.current.visible = true;
            } else {
              pinchIndicatorRef.current.visible = false;
            }

            // Only spawn relative to the cube being pinched (raycast to select anchor)
            if (isPinching && !handState.isDragging) {
              const mouse = new THREE.Vector2((indexTipMirrored.x - 0.5) * 2, -(indexTipMirrored.y - 0.5) * 2);
              const raycaster = new THREE.Raycaster();
              raycaster.setFromCamera(mouse, camera);
              const intersects = raycaster.intersectObjects(cubesRef.current);  // Raycast against all cubes
              if (intersects.length > 0) {
                handState.anchorCube = intersects[0].object;
                handState.lastSpawnPos = handState.anchorCube.position.clone();
                handState.pinchStartPos = new THREE.Vector3(
                  (indexTipMirrored.x - 0.5) * 10,
                  -(indexTipMirrored.y - 0.5) * 10,
                  0
                );
                handState.isDragging = true;

                // Optional highlight for anchor cube
                highlightRef.current.position.copy(handState.anchorCube.position);
                highlightRef.current.visible = true;
              }
            }

            // Compute drag delta, snap to grid in X/Y relative to anchor, only spawn if position changed
            if (isPinching && handState.isDragging) {
              const currentHandPos = new THREE.Vector3(
                (indexTipMirrored.x - 0.5) * 10,
                -(indexTipMirrored.y - 0.5) * 10,
                0
              );
              const deltaX = currentHandPos.x - handState.pinchStartPos.x;
              const deltaY = currentHandPos.y - handState.pinchStartPos.y;

              // Choose axis (strongest direction)
              const absDeltaX = Math.abs(deltaX);
              const absDeltaY = Math.abs(deltaY);
              let axis = null;
              let direction = 0;
              if (absDeltaX > absDeltaY && absDeltaX > dragThreshold) {
                axis = 'x';
                direction = deltaX > 0 ? 1 : -1;
              } else if (absDeltaY > dragThreshold) {
                axis = 'y';
                direction = deltaY > 0 ? 1 : -1;
              }

              if (axis) {
                // Snap to grid: Calculate snapped position relative to anchor
                const snappedDelta = Math.round((axis === 'x' ? deltaX : deltaY) / gridUnit) * gridUnit;
                const newBlockPos = handState.anchorCube.position.clone();
                newBlockPos[axis] += snappedDelta;
                newBlockPos.z = handState.anchorCube.position.z;  // Fixed Z

                // Only spawn if snapped position differs from last spawn
                if (!handState.lastSpawnPos.equals(newBlockPos)) {
                  const newBlock = new THREE.Mesh(
                    new THREE.BoxGeometry(1, 1, 1),
                    new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, opacity: 0.8 })
                  );
                  newBlock.position.copy(newBlockPos);
                  scene.add(newBlock);
                  cubesRef.current.push(newBlock);  // Add to cubes for future raycasting

                  const newOutline = new THREE.Mesh(outlineGeometry, outlineMaterial);
                  newBlock.add(outline);

                  handState.lastSpawnPos.copy(newBlockPos);
                  handState.pinchStartPos.copy(currentHandPos);
                }
              }
            }

            // Release logic (stop spawning, hide highlight)
            if (!isPinching && handState.isDragging) {
              handState.isDragging = false;
              handState.anchorCube = null;
              handState.lastSpawnPos = null;
              handState.pinchStartPos = null;
              highlightRef.current.visible = false;
            }
          });
        }
      }

      requestAnimationFrame(predictWebcam);
    };

    runHandLandmarker();
  }, []);

  return (
    <div style={{ textAlign: "center", position: "relative" }}>
      <h2>Refined Grid Snapping Relative to Pinched Cube</h2>
      <div id="three-container" style={{ position: "absolute", top: 50, left: "50%", transform: "translateX(-50%)" }}></div>
      <video ref={videoRef} autoPlay playsInline width="640" height="480" style={{ display: "none" }} />
      <canvas ref={canvasRef} width="640" height="480" />
      <p>Check console for logs. Red sphere shows pinch, yellow outline highlights pinched cube. Pinch any cube to select anchor, drag along X or Y to spawn blue blocks snapped to grid relative to anchor (Z fixed). Release to stop. Multi-hand supported.</p>
      <p>Thresholds: Pinch {pinchThreshold.toFixed(2)}, Drag {dragThreshold}, Grid {gridUnit} (adjust in code).</p>
    </div>
  );
}

export default App;