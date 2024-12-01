let model;
let labelMapping;
let hands;
let predictionHistory = [];
const historyLength = 8;
const confidenceThreshold = 0.6;
const minPredictionCount = 4;

async function loadModel() {
  try {
    model = await tf.loadLayersModel("/static/models/model.json");
    const response = await fetch("/static/models/label_mapping.json");
    labelMapping = await response.json();
    console.log("Model and labels loaded successfully!");
    document.getElementById("startButton").disabled = false;
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

async function setupCamera() {
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("output-canvas");

  // Set canvas size
  canvas.width = 640;
  canvas.height = 480;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 640,
        height: 480,
      },
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        resolve(video);
      };
    });
  } catch (error) {
    console.error("Error accessing camera:", error);
  }
}

function preprocessFrame(frame) {
  return tf.tidy(() => {
    // Convert to grayscale (average RGB channels)
    const grayscale = frame.mean(2).expandDims(2);

    // Normalize
    const normalized = grayscale.div(255.0);

    // Resize
    const resized = tf.image.resizeBilinear(normalized, [128, 128]);

    // Add batch dimension
    return resized.expandDims(0);
  });
}

async function detectHands(video) {
  const canvas = document.getElementById("output-canvas");
  const ctx = canvas.getContext("2d");

  // Initialize MediaPipe Hands
  if (!hands) {
    hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      },
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  }

  // Process the frame
  try {
    const results = await hands.process(video);

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        // Draw hand landmarks
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 5,
        });
        drawLandmarks(ctx, landmarks, { color: "#FF0000", lineWidth: 2 });

        // Get hand ROI
        const { x1, y1, x2, y2 } = getHandBoundingBox(landmarks, canvas);

        // Draw ROI box
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Extract ROI and make prediction
        const roi = ctx.getImageData(x1, y1, x2 - x1, y2 - y1);
        const tensor = preprocessFrame(tf.browser.fromPixels(roi));
        const prediction = await model.predict(tensor).data();
        console.log(prediction);
        // Get top 3 predictions
        const top3 = Array.from(prediction)
          .map((prob, idx) => ({ prob, label: labelMapping[idx] }))
          .sort((a, b) => b.prob - a.prob)
          .slice(0, 3);

        // Update predictions display
        updatePredictionsDisplay(top3, x1, y1);
      }
    }
  } catch (error) {
    console.error("Error in hand detection:", error);
  }

  requestAnimationFrame(() => detectHands(video));
}

function getHandBoundingBox(landmarks, canvas) {
  const width = canvas.width;
  const height = canvas.height;

  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;

  for (const landmark of landmarks) {
    const x = landmark.x * width;
    const y = landmark.y * height;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  // Add padding
  const padding = 40;
  return {
    x1: Math.max(0, minX - padding),
    y1: Math.max(0, minY - padding),
    x2: Math.min(width, maxX + padding),
    y2: Math.min(height, maxY + padding),
  };
}

function updatePredictionsDisplay(predictions, x, y) {
  const predsDiv = document.querySelector(".predictions");
  predsDiv.innerHTML = `
        <h3>Top 3 Predictions:</h3>
        ${predictions
          .map(
            (pred, i) => `
            <div class="prediction">
                ${i + 1}. ${pred.label}: ${(pred.prob * 100).toFixed(1)}%
            </div>
        `
          )
          .join("")}
    `;
}

async function startDetection() {
  const video = await setupCamera();
  if (video) {
    document.getElementById("startButton").disabled = true;
    document.getElementById("stopButton").disabled = false;
    detectHands(video);
  }
}

function stopDetection() {
  const video = document.getElementById("webcam");
  const stream = video.srcObject;
  const tracks = stream.getTracks();
  tracks.forEach((track) => track.stop());
  video.srcObject = null;

  document.getElementById("startButton").disabled = false;
  document.getElementById("stopButton").disabled = true;

  // Clear canvas
  const canvas = document.getElementById("output-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Clear predictions
  document.querySelector(".predictions").innerHTML = "";
}

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("startButton").disabled = true;
  document.getElementById("stopButton").disabled = true;

  document
    .getElementById("startButton")
    .addEventListener("click", startDetection);
  document
    .getElementById("stopButton")
    .addEventListener("click", stopDetection);

  loadModel();
});
