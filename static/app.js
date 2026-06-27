function setFeed() {
    const videoFeed = document.getElementById("videoFeed");
    videoFeed.src = "/video_feed?t=" + new Date().getTime();
}

function refreshFeed() {
    setFeed();
}

async function startRtsp() {
    const rtspUrl = document.getElementById("rtspUrl").value.trim();

    if (!rtspUrl) {
        alert("Please enter an RTSP URL.");
        return;
    }

    const response = await fetch("/api/start_rtsp", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            rtsp_url: rtspUrl
        })
    });

    const result = await response.json();

    if (!result.success) {
        alert(result.message);
        return;
    }

    setFeed();
}

async function uploadVideo() {
    const input = document.getElementById("videoFile");

    if (!input.files.length) {
        alert("Please select a video file.");
        return;
    }

    const formData = new FormData();
    formData.append("video", input.files[0]);

    const response = await fetch("/api/upload", {
        method: "POST",
        body: formData
    });

    const result = await response.json();

    if (!result.success) {
        alert(result.message);
        return;
    }

    setFeed();
}

async function stopProcessing() {
    await fetch("/api/stop", {
        method: "POST"
    });

    const videoFeed = document.getElementById("videoFeed");
    videoFeed.removeAttribute("src");

    updateStats();
}

async function updateStats() {
    try {
        const response = await fetch("/api/stats");
        const stats = await response.json();

        document.getElementById("status").innerText = stats.status;
        document.getElementById("currentCount").innerText = stats.current_count;
        document.getElementById("maxCount").innerText = stats.max_count;
        document.getElementById("fps").innerText = stats.fps;
        document.getElementById("frames").innerText = stats.total_frames;
    } catch (error) {
        console.error("Failed to update stats:", error);
    }
}

setInterval(updateStats, 1000);
updateStats();
