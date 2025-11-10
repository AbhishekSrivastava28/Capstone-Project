const recentTableBody = document.querySelector("#recent-table tbody");
const refreshBtn = document.getElementById("refresh-btn");
const refreshSpinner = document.getElementById("refresh-spinner");
const predictedMagnitude = document.getElementById("predicted-magnitude");
const predictedLocation = document.getElementById("predicted-location");
const predictionMethod = document.getElementById("prediction-method");
const previousList = document.getElementById("previous-list");
const lastRefreshed = document.getElementById("last-refreshed");
const globalLoader = document.getElementById("global-loader");
const precautionLevel = document.getElementById("precaution-level");
const precautionsList = document.getElementById("precautions-list");

let magnitudeChart = null;

const formatDate = (isoString) => {
    if (!isoString) {
        return "Unknown";
    }
    const date = new Date(isoString);
    return date.toLocaleString();
};

const toggleSpinner = (show) => {
    refreshSpinner.classList.toggle("d-none", !show);
    refreshBtn.disabled = show;
};

const toggleGlobalLoader = (show) => {
    globalLoader.classList.toggle("d-none", !show);
};

const renderRecentTable = (events) => {
    if (!events.length) {
        recentTableBody.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-4 text-muted">
                    No recent earthquake data.
                </td>
            </tr>
        `;
        return;
    }

    const rows = events
        .map(
            (event) => `
            <tr>
                <td>${event.place}</td>
                <td><span class="badge bg-danger-subtle text-danger-emphasis">${event.magnitude.toFixed(2)}</span></td>
                <td>${event.depth?.toFixed ? event.depth.toFixed(1) : event.depth}</td>
                <td>${formatDate(event.time)}</td>
            </tr>
        `
        )
        .join("");
    recentTableBody.innerHTML = rows;
};

const renderPrediction = (data) => {
    if (!data) return;
    const {
        predicted_magnitude,
        predicted_location,
        method,
        previous_events,
        precaution_level,
        precautions,
    } = data;

    predictedMagnitude.textContent = predicted_magnitude ? `M${predicted_magnitude}` : "--";
    predictedLocation.textContent = predicted_location || "Location unavailable";
    predictionMethod.textContent = method === "model" ? "Model Inference" : "Statistical Estimate";

    if (precautionLevel) {
        precautionLevel.textContent = precaution_level || "Guidance unavailable";
    }
    if (precautionsList) {
        if (precautions && precautions.length) {
            precautionsList.innerHTML = precautions
                .map(
                    (item) => `
                <li class="list-group-item">
                    <span class="fw-semibold text-success me-2">•</span>${item}
                </li>
            `
                )
                .join("");
        } else {
            precautionsList.innerHTML =
                '<li class="list-group-item text-muted text-center">Precaution guidance currently unavailable.</li>';
        }
    }

    if (!previous_events?.length) {
        previousList.innerHTML =
            '<li class="list-group-item text-muted text-center">No historical data available.</li>';
        if (magnitudeChart) {
            magnitudeChart.destroy();
            magnitudeChart = null;
        }
        return;
    }

    previousList.innerHTML = previous_events
        .map(
            (event) => `
            <li class="list-group-item d-flex flex-column">
                <div class="d-flex justify-content-between">
                    <span class="fw-semibold">${event.place}</span>
                    <span>M${Number(event.magnitude).toFixed(2)}</span>
                </div>
                <small class="text-muted">Depth: ${event.depth ? Number(event.depth).toFixed(1) : "?"} km · ${formatDate(event.time)}</small>
            </li>
        `
        )
        .join("");

    const labels = previous_events.map((event) => event.place.split(",")[0]);
    const dataPoints = previous_events.map((event) => parseFloat(Number(event.magnitude).toFixed(2)));

    const ctx = document.getElementById("magnitude-chart");
    if (magnitudeChart) {
        magnitudeChart.destroy();
    }
    magnitudeChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Magnitude",
                    data: dataPoints,
                    borderColor: "#3f51b5",
                    backgroundColor: "rgba(63, 81, 181, 0.2)",
                    lineTension: 0.25,
                    fill: true,
                },
            ],
        },
        options: {
            scales: {
                y: {
                    suggestedMin: 3,
                    suggestedMax: Math.max(...dataPoints) + 0.5,
                    title: {
                        display: true,
                        text: "Magnitude",
                    },
                },
            },
            plugins: {
                legend: {
                    display: false,
                },
            },
        },
    });
};

const fetchRecent = async () => {
    try {
        const response = await fetch("/recent");
        if (!response.ok) {
            throw new Error("Failed to fetch recent data.");
        }
        const data = await response.json();
        renderRecentTable(data.events || []);
        if (lastRefreshed) {
            lastRefreshed.textContent = data.last_refreshed
                ? `Last refreshed: ${formatDate(data.last_refreshed)}`
                : "";
        }
    } catch (error) {
        console.error(error);
        recentTableBody.innerHTML = `
            <tr>
                <td colspan="4" class="text-center py-4 text-danger">
                    Unable to load recent earthquake data.
                </td>
            </tr>
        `;
    }
};

const fetchPrediction = async () => {
    try {
        const response = await fetch("/predict");
        if (!response.ok) {
            throw new Error("Failed to fetch prediction.");
        }
        const data = await response.json();
        renderPrediction(data);
    } catch (error) {
        console.error(error);
    }
};

const refreshData = async () => {
    toggleSpinner(true);
    try {
        const response = await fetch("/refresh", { method: "POST" });
        if (!response.ok) {
            throw new Error("Refresh request failed");
        }
        const data = await response.json();
        renderRecentTable(data.events || []);
        if (lastRefreshed) {
            lastRefreshed.textContent = data.last_refreshed
                ? `Last refreshed: ${formatDate(data.last_refreshed)}`
                : "";
        }
        await fetchPrediction();
    } catch (error) {
        console.error(error);
    } finally {
        toggleSpinner(false);
    }
};

const initialiseDashboard = async () => {
    toggleGlobalLoader(true);
    await Promise.all([fetchRecent(), fetchPrediction()]);
    toggleGlobalLoader(false);
};

refreshBtn?.addEventListener("click", refreshData);

// Auto refresh every 10 minutes
setInterval(refreshData, 10 * 60 * 1000);

initialiseDashboard();

