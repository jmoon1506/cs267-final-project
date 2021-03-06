var theChart = new Chart(document.getElementById("perf"), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { 
        data: [],
        label: "Serial",
        borderColor: "#75B9BE",
        backgroundColor: "#75B9BE",
        fill: false
      },
      { 
        data: [],
        label: "Shared memory",
        borderColor: "#EE7674",
        backgroundColor: "#EE7674",
        fill: false
      },
      { 
        data: [],
        label: "Distributed memory",
        borderColor: "#A6B1E1",
        backgroundColor: "#A6B1E1",
        fill: false
      },
    ]
  },
  options: {
    responsive: true,
    title: {
      display: true,
      text: 'Performance',
    },
    legend: {
      display: true,
      position: 'bottom',
      labels: {
        boxWidth: 20,
      },
      // reverse: true,
    },
    tooltips: {
      enabled: true,
      mode: 'index',
      callbacks: {
        title: function(tooltipItem) {
          console.log(tooltipItem);
          return "Turn " + (tooltipItem[0].index+1);
        },
        label: function(tooltipItem) {
          return Number(tooltipItem.yLabel).toFixed(2) + " seconds";
        }
      }
    },
    scales: {
      xAxes: [{
        // stacked: true,
        ticks: {
          beginAtZero: true,
          autoSkip: true,
          autoSkipPadding: 20,
        },
        scaleLabel: {
          display: true,
          labelString: 'Turn'
        }
      }],
      yAxes: [{
        // stacked: true,
        ticks: {
          beginAtZero: true,
        },
        scaleLabel: {
          display: true,
          labelString: 'Compute time'
        }
      }],
    }
  }
});

function hover_perf(e) {
  theChart.options.tooltips.enabled = true;
};

function unhover_perf(e) {
  theChart.options.tooltips.enabled = false;
};