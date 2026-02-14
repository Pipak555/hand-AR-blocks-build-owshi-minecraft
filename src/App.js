import { useRef, useEffect } from "react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as THREE from "three";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const cubeRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const startedRef = useRef(false);
  const isGrabbingRef = useRef(false);

  // Tunable thresholds (adjust these for your setup)
  const pinchThreshold = 0.05;  // Distance between thumb & index to register pinch
  const grabThreshold = 0.15;   // Distance from hand to cube to allow grabbing

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

    // Create cube with subtle outline for "grabbable" feedback
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.8 });
    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);
    cubeRef.current = cube;

    // Add subtle outline (wireframe) to indicate grabbable state
    const outlineGeometry = new THREE.BoxGeometry(1.05, 1.05, 1.05);
    const outlineMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true });
    const outline = new THREE.Mesh(outlineGeometry, outlineMaterial);
    cube.add(outline);  // Attach to cube

    // Append renderer to DOM
    const container = document.getElementById("three-container");
    if (container) container.appendChild(renderer.domElement);

    // Animate Three.js
    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // Step 1: Enable two-hand detection
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
        numHands: 2,  // Detect up to 2 hands
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

      // Step 3: Apply camera inversion (flip horizontally for natural interaction)
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.scale(-1, 1);  // Horizontal flip
      ctx.drawImage(videoRef.current, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      if (handLandmarkerRef.current && videoRef.current) {
        const results = handLandmarkerRef.current.detectForVideo(videoRef.current, Date.now());

        let closestHand = null;
        let minDistance = Infinity;

        // Step 2: Update rendering loop for both hands
        if (results.landmarks) {
          results.landmarks.forEach((landmarks, index) => {
            // Assign colors: Hand 1 → red, Hand 2 → blue
            const color = index === 0 ? "red" : "blue";
            ctx.fillStyle = color;
            ctx.strokeStyle = color;

            // Draw hand landmarks (dots and connections)
            for (const landmark of landmarks) {
              ctx.beginPath();
              ctx.arc((1 - landmark.x) * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);  // Mirror X for flipped canvas
              ctx.fill();
            }

            // Optional label
            const wrist = landmarks[0];
            ctx.fillText(`Hand ${index + 1}`, (1 - wrist.x) * canvas.width, wrist.y * canvas.height - 20);

            // Get mirrored key points for grab logic
            const indexTip = { x: 1 - landmarks[8].x, y: landmarks[8].y };  // Mirror X
            const thumbTip = { x: 1 - landmarks[4].x, y: landmarks[4].y };  // Mirror X
            const wristMirrored = { x: 1 - landmarks[0].x, y: landmarks[0].y };  // Mirror X

            // Check pinch
            const pinchDistance = Math.sqrt(
              Math.pow(indexTip.x - thumbTip.x, 2) + Math.pow(indexTip.y - thumbTip.y, 2)
            );
            const isPinching = pinchDistance < pinchThreshold;

            // Check proximity to cube (using mirrored coordinates)
            const cubeX = (cube.position.x / 10) + 0.5;
            const cubeY = -(cube.position.y / 10) + 0.5;
            const handToCubeDistance = Math.sqrt(
              Math.pow(indexTip.x - cubeX, 2) + Math.pow(indexTip.y - cubeY, 2)
            );
            const canGrab = handToCubeDistance < grabThreshold;

            // Step 4: Adjust grab logic for two hands (choose closest pinching hand)
            if (isPinching && canGrab && handToCubeDistance < minDistance) {
              minDistance = handToCubeDistance;
              closestHand = { indexTip, wrist: wristMirrored };
            }
          });
        }

        // Set isGrabbing based on closest hand
        if (closestHand && !isGrabbingRef.current) {
          isGrabbingRef.current = true;
        } else if (!closestHand) {
          isGrabbingRef.current = false;
        }

        // Step 3 & 4: Update cube only if grabbed (using mirrored coordinates)
        if (isGrabbingRef.current && closestHand) {
          cube.position.x = (closestHand.indexTip.x - 0.5) * 10;
          cube.position.y = -(closestHand.indexTip.y - 0.5) * 10;
          cube.rotation.z = (closestHand.wrist.x - 0.5) * Math.PI;
          cube.material.color.setHex(0xff0000);  // Red when grabbed
        } else {
          cube.material.color.setHex(0x00ff00);  // Green when released
        }
      }

      requestAnimationFrame(predictWebcam);
    };

    runHandLandmarker();
  }, []);

  return (
    <div style={{ textAlign: "center", position: "relative" }}>
      <h2>Two-Hand Tracked 3D Cube (Mirrored Camera)</h2>
      <div id="three-container" style={{ position: "absolute", top: 50, left: "50%", transform: "translateX(-50%)" }}></div>
      <video ref={videoRef} autoPlay playsInline width="640" height="480" style={{ display: "none" }} />
      <canvas ref={canvasRef} width="640" height="480" />
      <p>Put both hands in frame: Red dots for Hand 1, Blue for Hand 2. Pinch near cube with closest hand to grab (red), move/rotate. Open to release (green). Mirrored for natural movement!</p>
      <p>Thresholds: Pinch threshold {pinchThreshold.toFixed(2)}, Grab threshold {grabThreshold.toFixed(2)} (adjust in code if needed).</p>
    </div>
  );
}

export default App;