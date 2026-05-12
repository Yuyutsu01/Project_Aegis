/* 
    AEGIS V2 | DASHBOARD LOGIC
    Refined version with robust logging and stable charting.
*/

console.log("Aegis app.js loaded.");

window.addEventListener('load', () => {
    console.log("Window loaded, initializing Aegis...");
    
    let rawData = [];
    let equityChart, drawdownChart;
    let equitySeries, drawdownSeries;

    function initCharts() {
        console.log("Initializing charts...");
        const chartOptions = {
            layout: {
                backgroundColor: '#161B22',
                textColor: '#8B949E',
            },
            grid: {
                vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
                horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
            },
            crosshair: {
                mode: 0,
            },
            timeScale: {
                borderColor: 'rgba(255, 255, 255, 0.1)',
            },
        };

        const equityContainer = document.getElementById('equity-chart');
        const drawdownContainer = document.getElementById('drawdown-chart');

        if (!equityContainer || !drawdownContainer) {
            console.error("Chart containers not found!");
            return;
        }

        equityChart = LightweightCharts.createChart(equityContainer, {
            ...chartOptions,
            height: 400,
        });
        
        equitySeries = equityChart.addAreaSeries({
            topColor: 'rgba(0, 209, 255, 0.4)',
            bottomColor: 'rgba(0, 209, 255, 0.0)',
            lineColor: '#00D1FF',
            lineWidth: 2,
        });

        drawdownChart = LightweightCharts.createChart(drawdownContainer, {
            ...chartOptions,
            height: 150,
        });
        
        drawdownSeries = drawdownChart.addAreaSeries({
            topColor: 'rgba(255, 77, 77, 0.4)',
            bottomColor: 'rgba(255, 77, 77, 0.0)',
            lineColor: '#FF4D4D',
            lineWidth: 1,
        });

        window.addEventListener('resize', () => {
            equityChart.resize(equityContainer.clientWidth, 400);
            drawdownChart.resize(drawdownContainer.clientWidth, 150);
        });
        
        console.log("Charts initialized successfully.");
    }

    async function fetchData() {
        console.log("Fetching data from /api/data...");
        try {
            const response = await fetch('/api/data');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            rawData = await response.json();
            console.log(`Data received: ${rawData.length} records.`);
            
            if (rawData.length > 0) {
                updateDashboard('Full Backtest');
            } else {
                console.warn("Received empty dataset.");
            }
        } catch (error) {
            console.error("Fetch failed:", error);
            document.getElementById('page-subtitle').innerText = `Sync Error: ${error.message}`;
            document.getElementById('page-subtitle').style.color = '#FF4D4D';
        }
    }

    function updateDashboard(period) {
        console.log(`Updating dashboard for period: ${period}`);
        let filtered = rawData;
        if (period === 'XGB Train') filtered = rawData.filter(d => d.period === 'IS_XGB_TRAIN');
        if (period === 'PPO Train') filtered = rawData.filter(d => d.period === 'IS_PPO_TRAIN');
        if (period === 'OOS Test') filtered = rawData.filter(d => d.period === 'OOS_TEST');

        if (filtered.length === 0) {
            console.warn("No data for selected period.");
            return;
        }

        const first = filtered[0];
        const last = filtered[filtered.length - 1];
        
        // Metrics
        const totalRet = (last.equity / first.equity) - 1;
        document.getElementById('metric-liquidity').innerText = `$ ${(last.equity * 10000).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        
        const trendEl = document.getElementById('metric-return-trend');
        trendEl.innerHTML = `<i class="fa-solid fa-arrow-trend-${totalRet >= 0 ? 'up' : 'down'}"></i> <span>${(totalRet * 100).toFixed(2)}%</span>`;
        trendEl.className = `metric-trend trend-${totalRet >= 0 ? 'up' : 'down'}`;

        // Sharpe (subset)
        const pnls = filtered.map(d => d.pnl);
        const meanPnl = pnls.reduce((a, b) => a + b, 0) / pnls.length;
        const stdPnl = Math.sqrt(pnls.map(x => Math.pow(x - meanPnl, 2)).reduce((a, b) => a + b, 0) / pnls.length);
        const sharpe = (meanPnl / stdPnl * Math.sqrt(252)) || 0;
        document.getElementById('metric-sharpe').innerText = sharpe.toFixed(2);

        // Max DD
        let currMax = -Infinity;
        const dds = filtered.map(d => {
            if (d.equity > currMax) currMax = d.equity;
            return (d.equity / currMax) - 1;
        });
        const maxDD = Math.min(...dds);
        document.getElementById('metric-drawdown').innerText = `${(maxDD * 100).toFixed(2)}%`;

        // Win Rate
        const winRate = (pnls.filter(p => p > 0).length / pnls.length) * 100;
        document.getElementById('metric-winrate').innerText = `${winRate.toFixed(1)}%`;

        // Chart Data
        const equityData = filtered.map((d, i) => ({
            time: i,
            value: (d.equity / first.equity) * 10000
        }));
        const drawdownData = dds.map((val, i) => ({
            time: i,
            value: val
        }));

        equitySeries.setData(equityData);
        drawdownSeries.setData(drawdownData);

        // Intel Panel
        const lastRec = filtered[filtered.length - 1];
        const xgbConf = Math.abs(lastRec.xgb_prob - 0.5) * 200;
        document.getElementById('intel-xgb-val').innerText = `${xgbConf.toFixed(0)}%`;
        document.getElementById('intel-xgb-bar').style.width = `${xgbConf}%`;

        const rlAlign = lastRec.position !== 0 ? 100 : 0;
        document.getElementById('intel-rl-val').innerText = `${rlAlign}%`;
        document.getElementById('intel-rl-bar').style.width = `${rlAlign}%`;

        const stance = document.getElementById('strategy-stance');
        if (lastRec.position > 0) {
            stance.innerText = 'BULLISH / LONG';
            stance.className = 'badge badge-long';
        } else if (lastRec.position < 0) {
            stance.innerText = 'BEARISH / SHORT';
            stance.className = 'badge badge-short';
        } else {
            stance.innerText = 'NEUTRAL / CASH';
            stance.className = 'badge badge-neutral';
        }

        updateLedger(filtered);
    }

    function updateLedger(data) {
        const body = document.getElementById('ledger-body');
        body.innerHTML = '';
        const displayData = data.slice(-50).reverse();

        displayData.forEach(d => {
            const row = document.createElement('tr');
            row.className = 'row-hover';
            
            let statusBadge = '';
            if (d.position > 0) statusBadge = '<span class="badge badge-long">Long</span>';
            else if (d.position < 0) statusBadge = '<span class="badge badge-short">Short</span>';
            else statusBadge = '<span class="badge badge-neutral">Neutral</span>';

            row.innerHTML = `
                <td>${d.date}</td>
                <td style="font-weight: 700;">${d.position > 0 ? 'BUY' : d.position < 0 ? 'SELL' : 'HOLD'}</td>
                <td style="font-family: monospace;">${d.xgb_prob.toFixed(4)}</td>
                <td style="font-family: monospace;" class="${d.pnl >= 0 ? 'trend-up' : 'trend-down'}">${(d.pnl * 100).toFixed(3)}%</td>
                <td style="font-family: monospace;">$ ${(d.equity * 10000).toFixed(2)}</td>
                <td>${statusBadge}</td>
            `;
            body.appendChild(row);
        });
    }

    // Filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            updateDashboard(btn.getAttribute('data-period'));
        });
    });

    initCharts();
    fetchData();
});
