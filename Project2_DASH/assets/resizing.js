window.addEventListener('resize', function() {
    Plotly.Plots.resize(document.getElementById('scatter-plot'));
    Plotly.Plots.resize(document.getElementById('nutrition-correlation-heatmap'));
    Plotly.Plots.resize(document.getElementById('boxplot-1'));
    Plotly.Plots.resize(document.getElementById('boxplot-2'));
    Plotly.Plots.resize(document.getElementById('heatmap'));
});
