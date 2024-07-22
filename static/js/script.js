document.addEventListener('DOMContentLoaded', () => {
    const dataServiceUrl = '/fetch-data';  // Adjust URL if needed
    const processServiceUrl = '/process-data';  // Adjust URL if needed
    const forecastServiceUrl = '/generate-forecast';  // Adjust URL if needed

    // Helper functions
    function updateProgressBar(id, percentage) {
        const progressBar = document.getElementById(id);
        progressBar.style.width = percentage + '%';
        progressBar.textContent = percentage + '%';
    }

    function showStatus(id, message) {
        document.getElementById(id).textContent = message;
        document.getElementById(id).style.display = 'block';
    }

    function hideStatus(id) {
        document.getElementById(id).style.display = 'none';
    }

    function showError(id, message) {
        document.getElementById(id).textContent = message;
        document.getElementById(id).style.display = 'block';
    }

    function hideError(id) {
        document.getElementById(id).style.display = 'none';
    }

    // Fetch Data
    document.getElementById('fetch-data').addEventListener('click', async () => {
        hideError('data-error');
        showStatus('data-status', 'Loading data...');
        updateProgressBar('data-progress', 0);
        try {
            const response = await fetch(dataServiceUrl, { method: 'POST' });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            document.getElementById('data-output').textContent = JSON.stringify(data, null, 2);
            updateProgressBar('data-progress', 100);
            hideStatus('data-status');
            const coinSelect = document.getElementById('coin-select');
            coinSelect.innerHTML = '<option value="">Select a coin</option>';
            for (const coin of data.coins) {
                const option = document.createElement('option');
                option.value = coin.symbol;
                option.textContent = coin.name;
                coinSelect.appendChild(option);
            }
        } catch (error) {
            console.error('Fetch data error:', error);  // Log detailed error
            showError('data-error', 'Error fetching data: ' + error.message);
            updateProgressBar('data-progress', 0);
            hideStatus('data-status');
        }
    });

    // Process Data
    document.getElementById('process-data').addEventListener('click', async () => {
        hideError('process-error');
        showStatus('process-status', 'Processing data...');
        updateProgressBar('process-progress', 0);
        try {
            const rawData = document.getElementById('data-output').textContent;
            const response = await fetch(processServiceUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: rawData
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const processedData = await response.json();
            document.getElementById('processed-data-output').textContent = JSON.stringify(processedData, null, 2);
            updateProgressBar('process-progress', 100);
            hideStatus('process-status');
        } catch (error) {
            console.error('Process data error:', error);  // Log detailed error
            showError('process-error', 'Error processing data: ' + error.message);
            updateProgressBar('process-progress', 0);
            hideStatus('process-status');
        }
    });

    // Generate Forecast
    document.getElementById('forecast-data').addEventListener('click', async () => {
        hideError('forecast-error');
        showStatus('forecast-status', 'Generating forecast...');
        const coin = document.getElementById('coin-select').value;
        if (!coin) {
            showError('forecast-error', 'Please select a coin.');
            hideStatus('forecast-status');
            return;
        }
        try {
            const processedData = document.getElementById('processed-data-output').textContent;
            const response = await fetch(forecastServiceUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ coin, data: JSON.parse(processedData) })
            });
            if (!response.ok) throw new Error('Network response was not ok');
            const result = await response.json();
            document.getElementById('forecast-image').src = `data:image/png;base64,${result.image}`;
            hideStatus('forecast-status');
        } catch (error) {
            showError('forecast-error', 'Error generating forecast: ' + error.message);
            hideStatus('forecast-status');
        }
    });
});
