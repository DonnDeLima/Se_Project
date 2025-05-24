// script.js

const cameraFeed = document.getElementById("cameraFeed");
const registerButton = document.getElementById("registerButton");
const authenticateButton = document.getElementById("authenticateButton");

// Start camera feed
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        cameraFeed.srcObject = stream;
    })
    .catch((error) => {
        console.error("Error accessing webcam:", error);
    });

// Function to capture an image from the video feed
function captureImage() {
    const canvas = document.createElement("canvas");
    canvas.width = cameraFeed.videoWidth;
    canvas.height = cameraFeed.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/png"); // Returns the image as a Base64 string
}

// Handle registration
registerButton.addEventListener("click", () => {
    const userName = prompt("Enter your name:");
    if (!userName) return alert("Registration canceled.");

    const imageData = captureImage();
    fetch("/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: userName, image: imageData }),
    })
        .then((response) => response.json())
        .then((data) => alert(data.message))
        .catch((error) => console.error("Error during registration:", error));
});

// Handle authentication
authenticateButton.addEventListener("click", () => {
    const imageData = captureImage();
    fetch("/authenticate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
    })
        .then((response) => response.json())
        .then((data) => alert(data.message))
        .catch((error) => console.error("Error during authentication:", error));
});
