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

    // Create cube
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);
    cubeRef.current = cube;

    // Append renderer to DOM
    const container = document.getElementById("three-container");
    if (container) container.appendChild(renderer.domElement);

    // Animate Three.js
    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // Setup hand tracking
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
        numHands: 1,
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

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      if (handLandmarkerRef.current && videoRef.current) {
        const results = handLandmarkerRef.current.detectForVideo(videoRef.current, Date.now());

        if (results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0];

          // Draw hand dots
          ctx.fillStyle = "red";
          for (const landmark of landmarks) {
            ctx.beginPath();
            ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
            ctx.fill();
          }

          // Get key points
          const indexTip = landmarks[8];  // Index finger tip
          const thumbTip = landmarks[4];  // Thumb tip
          const wrist = landmarks[0];  // Wrist

          // Pinch detection (grab/release)
          const distance = Math.sqrt(
            Math.pow(indexTip.x - thumbTip.x, 2) + Math.pow(indexTip.y - thumbTip.y, 2)
          );
          const pinchThreshold = 0.05;  // Adjust for sensitivity
          isGrabbingRef.current = distance < pinchThreshold;

          // Update cube if grabbing
          if (isGrabbingRef.current) {
            cube.position.x = (indexTip.x - 0.5) * 10;  // Map to scene (-5 to 5)
            cube.position.y = -(indexTip.y - 0.5) * 10;
            cube.rotation.z = (wrist.x - 0.5) * Math.PI;  // Rotate based on wrist
            cube.material.color.setHex(0xff0000);  // Red when grabbed
          } else {
            cube.material.color.setHex(0x00ff00);  // Green when released
          }
        }
      }

      requestAnimationFrame(predictWebcam);
    };

    runHandLandmarker();
  }, []);

  return (
    <div style={{ textAlign: "center", position: "relative" }}>
      <h2>Hand-Tracked 3D Cube</h2>
      <div id="three-container" style={{ position: "absolute", top: 50, left: "50%", transform: "translateX(-50%)" }}></div>
      <video ref={videoRef} autoPlay playsInline width="640" height="480" style={{ display: "none" }} />
      <canvas ref={canvasRef} width="640" height="480" />
      <p>Pinch (thumb + index) to grab/move/rotate the cube. Open hand to release.</p>
    </div>
  );
}

export default App;