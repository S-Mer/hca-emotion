// Timeline configuration
const config = {
  margin: { top: 100, right: 60, bottom: 80, left: 60 },
  verticalSpacing: 25, // Space between stacked tales
  circleRadius: 8,
  maxCircleRadius: 20,
};

// Emotion color palette
const emotionColors = {
  sadness: "#4A90E2",
  joy: "#7ED321",
  anger: "#D0021B",
  fear: "#9013FE",
  disgust: "#F5A623",
  surprise: "#F8E71C",
  neutral: "#9B9B9B",
};

// Sort order for emotions (for visual grouping)
// From timeline upward: disgust -> fear -> sadness -> anger -> neutral -> surprise -> joy
const emotionOrder = {
  disgust: 1,
  fear: 2,
  sadness: 3,
  anger: 4,
  neutral: 5,
  surprise: 6,
  joy: 7,
};

// Global variables for search functionality
let taleGroups;
let g;
let talesWithPositions;

function createAbbreviatedTimeScale(width) {
  // Define the time segments we want to show
  // Visual spacing: make each 15-year period the same width (80px)
  const segmentWidth = 80;
  const gapWidth = 50; // Space for zigzag breaks

  const timeSegments = [
    {
      period: "birth",
      start: 1805,
      end: 1805,
      displayStart: 0,
      displayEnd: segmentWidth,
      yearMarkers: [], // No year marker - will use Birth life event instead
    },
    {
      period: "early",
      start: 1820,
      end: 1834,
      displayStart: segmentWidth + gapWidth,
      displayEnd: segmentWidth + gapWidth + segmentWidth,
      yearMarkers: [{ year: 1820, displayX: segmentWidth + gapWidth }],
    },
    {
      period: "productive",
      start: 1835,
      end: 1875,
      displayStart: 2 * segmentWidth + 2 * gapWidth,
      displayEnd: width,
      yearMarkers: [], // Will be calculated dynamically
    },
  ];

  // Create mapping functions
  function yearToDisplay(year) {
    for (let segment of timeSegments) {
      if (year >= segment.start && year <= segment.end) {
        const segmentProgress =
          (year - segment.start) / (segment.end - segment.start);
        return (
          segment.displayStart +
          segmentProgress * (segment.displayEnd - segment.displayStart)
        );
      }
    }
    return null; // Year not in any segment
  }

  return { yearToDisplay, timeSegments };
}

function drawTimelineWithBreaks(g, timeSegments, timelineY, height) {
  // Calculate year markers for the productive period
  const productiveSegment = timeSegments[2];
  const yearInterval = 5; // Show every 5 years

  for (let year = 1835; year <= 1875; year += yearInterval) {
    const progress =
      (year - productiveSegment.start) /
      (productiveSegment.end - productiveSegment.start);
    const segmentLength =
      productiveSegment.displayEnd - productiveSegment.displayStart;
    const displayX = productiveSegment.displayStart + progress * segmentLength;
    productiveSegment.yearMarkers.push({ year, displayX });
  }

  // Draw each timeline segment
  timeSegments.forEach((segment) => {
    // Main timeline line
    g.append("line")
      .attr("class", "timeline-segment")
      .attr("x1", segment.displayStart)
      .attr("x2", segment.displayEnd)
      .attr("y1", timelineY)
      .attr("y2", timelineY)
      .attr("stroke", "#666")
      .attr("stroke-width", 3);

    // Draw year markers with vertical lines
    segment.yearMarkers.forEach((marker) => {
      // Vertical indicator line
      g.append("line")
        .attr("x1", marker.displayX)
        .attr("x2", marker.displayX)
        .attr("y1", timelineY - 10)
        .attr("y2", timelineY + 10)
        .attr("stroke", "#555")
        .attr("stroke-width", 1.5);

      // Year label
      g.append("text")
        .attr("x", marker.displayX)
        .attr("y", timelineY + 30)
        .attr("text-anchor", "middle")
        .attr("fill", "#aaa")
        .attr("font-size", "11px")
        .attr("font-weight", "bold")
        .text(marker.year);
    });
  });

  // Draw break indicators
  for (let i = 0; i < timeSegments.length - 1; i++) {
    const currentEnd = timeSegments[i].displayEnd;
    const nextStart = timeSegments[i + 1].displayStart;
    const gapWidth = nextStart - currentEnd;
    const breakCenter = currentEnd + gapWidth / 2;

    // Calculate zigzag dimensions symmetrically
    const zigzagWidth = gapWidth * 0.6; // Use 60% of gap for zigzag
    const zigzagStart = breakCenter - zigzagWidth / 2;
    const zigzagEnd = breakCenter + zigzagWidth / 2;
    const zigHeight = 10;

    // Zigzag break symbol - perfectly centered
    const zigzag = `M ${zigzagStart},${timelineY - zigHeight}
                   L ${breakCenter - zigzagWidth / 6},${timelineY + zigHeight}
                   L ${breakCenter},${timelineY - zigHeight}
                   L ${breakCenter + zigzagWidth / 6},${timelineY + zigHeight}
                   L ${zigzagEnd},${timelineY - zigHeight}`;

    g.append("path")
      .attr("d", zigzag)
      .attr("stroke", "#666")
      .attr("stroke-width", 2)
      .attr("fill", "none");

    // Break label
    g.append("text")
      .attr("x", breakCenter)
      .attr("y", timelineY + 30)
      .attr("text-anchor", "middle")
      .attr("fill", "#888")
      .attr("font-size", "10px")
      .attr("font-style", "italic")
      .text("...");
  }

  // Add major life events at correct positions
  const lifeEvents = [
    {
      year: 1805,
      label: "Birth",
      shape: "cross",
      color: "#888",
      displayX: timeSegments[0].displayStart, // Start of timeline
    },
    {
      year: 1875,
      label: "Death",
      shape: "cross",
      color: "#888",
      displayX: timeSegments[2].displayEnd, // End of timeline
    },
  ];

  lifeEvents.forEach((event) => {
    // Cross for birth and death
    const crossSize = 7;
    g.append("path")
      .attr(
        "d",
        `M ${event.displayX - crossSize},${timelineY} L ${
          event.displayX + crossSize
        },${timelineY} M ${event.displayX},${timelineY - crossSize} L ${
          event.displayX
        },${timelineY + crossSize}`
      )
      .attr("stroke", event.color)
      .attr("stroke-width", 3)
      .attr("stroke-linecap", "round");

    // Label
    g.append("text")
      .attr("x", event.displayX)
      .attr("y", timelineY - 15)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .attr("font-size", "11px")
      .attr("font-weight", "bold")
      .text(event.label);
  });
}

async function init() {
  // Load data
  const rawData = await d3.json("data/timeline_data.json");

  const data = rawData
    .map((d) => ({
      ...d,
      year: +d.year,
      num_chunks: +d.num_chunks,
      sadness: +d.sadness,
      joy: +d.joy,
      fear: +d.fear,
      anger: +d.anger,
      disgust: +d.disgust,
      surprise: +d.surprise,
      neutral: +d.neutral,
      emotion_purity: +d.emotion_purity,
    }))
    .filter((d) => !isNaN(d.year) && d.year > 0);

  console.log("Cleaned data:", data.length, "tales");

  // Set up SVG dimensions
  const container = d3.select("#timeline-container");
  const containerWidth = container.node().getBoundingClientRect().width;

  const width = containerWidth - config.margin.left - config.margin.right;
  const height = 800 - config.margin.top - config.margin.bottom;
  const timelineY = height; // TIMELINE AT BOTTOM

  const svg = d3
    .select("#timeline")
    .attr("width", containerWidth)
    .attr("height", 800);

  g = svg
    .append("g")
    .attr(
      "transform",
      `translate(${config.margin.left}, ${config.margin.top})`
    );

  // Create abbreviated scale
  const { yearToDisplay, timeSegments } = createAbbreviatedTimeScale(width);

  // Filter data to only include years in our segments
  const validData = data.filter((d) => yearToDisplay(d.year) !== null);

  const sizeScale = d3
    .scaleSqrt()
    .domain(d3.extent(validData, (d) => d.num_chunks))
    .range([config.circleRadius, config.maxCircleRadius]);

  // Add biographical context
  // addBiographicalContext(g, timeSegments, timelineY, width, height);

  // Draw main timeline with breaks
  drawTimelineWithBreaks(g, timeSegments, timelineY, height);

  // Group tales by display position
  const talesByDisplayX = d3.group(validData, (d) =>
    Math.round(yearToDisplay(d.year))
  );

  // Calculate positions with smart sorting
  talesWithPositions = [];

  talesByDisplayX.forEach((tales, displayX) => {
    // SORT TALES WITHIN EACH YEAR
    // Order from timeline upward: disgust -> fear -> sadness -> anger -> neutral -> surprise -> joy
    const sortedTales = tales.sort((a, b) => {
      // 1. Group by emotion (primary sort)
      const emotionDiff =
        emotionOrder[a.dominant_emotion] - emotionOrder[b.dominant_emotion];
      if (emotionDiff !== 0) return emotionDiff;

      // 2. Certainty (secondary sort) - confirmed dates before approximate
      if (a.approximate !== b.approximate) {
        return a.approximate ? 1 : -1;
      }

      // 3. Size (tertiary sort) - smaller tales first
      return a.num_chunks - b.num_chunks;
    });

    // Calculate vertical positions (stack upward from timeline)
    sortedTales.forEach((tale, index) => {
      const taleSize = sizeScale(tale.num_chunks);

      let cumulativeHeight = 0;
      for (let i = 0; i < index; i++) {
        const prevSize = sizeScale(sortedTales[i].num_chunks);
        cumulativeHeight += prevSize * 2 + config.verticalSpacing;
      }

      tale.y = timelineY - (taleSize + cumulativeHeight + 10); // Stack upward
      tale.x = displayX;
      tale.size = taleSize;

      talesWithPositions.push(tale);
    });
  });

  // Find max height for scaling if needed
  const maxHeight = Math.abs(d3.min(talesWithPositions, (d) => d.y));
  console.log("Max stack height:", maxHeight);

  // Draw connecting lines (from timeline up to tale)
  g.selectAll(".stack-line")
    .data(talesWithPositions)
    .join("line")
    .attr("class", "stack-line")
    .attr("x1", (d) => d.x)
    .attr("y1", timelineY)
    .attr("x2", (d) => d.x)
    .attr("y2", (d) => d.y)
    .attr("stroke", "#444")
    .attr("stroke-width", 1)
    .attr("opacity", 0.2);

  // Create tooltip
  const tooltip = d3.select("#tooltip");

  // Draw tale groups
  taleGroups = g
    .selectAll(".tale-group")
    .data(talesWithPositions)
    .join("g")
    .attr("class", "tale-group")
    .attr("transform", (d) => `translate(${d.x}, ${d.y})`);

  // Draw shapes
  taleGroups.each(function (d) {
    const group = d3.select(this);
    const color = emotionColors[d.dominant_emotion] || "#999";
    const opacity = calculateOpacity(d.emotion_purity);

    if (d.approximate) {
      // Diamond for approximate
      const points = [
        [0, -d.size],
        [d.size, 0],
        [0, d.size],
        [-d.size, 0],
      ];

      group
        .append("polygon")
        .attr("class", "tale-shape")
        .attr("points", points.map((p) => p.join(",")).join(" "))
        .style("fill", color)
        .style("opacity", opacity)
        .style("stroke", "none");
    } else {
      // Circle for confirmed
      group
        .append("circle")
        .attr("class", "tale-shape")
        .attr("r", d.size)
        .style("fill", color)
        .style("opacity", opacity)
        .style("stroke", "none");
    }
  });

  // Interactions
  taleGroups
    .on("mouseover", function (event, d) {
      // Highlight this tale
      d3.select(this)
        .select(".tale-shape")
        .style("stroke", "#ffffff")
        .style("stroke-width", 3);

      // Dim other tales
      taleGroups.style("opacity", 0.3);
      d3.select(this).style("opacity", 1);

      // Highlight the connecting line
      g.selectAll(".stack-line").attr("opacity", (line) =>
        line.tale === d.tale ? 0.6 : 0.1
      );

      showTooltip(event, d, tooltip);
    })
    .on("mouseout", function () {
      d3.select(this).select(".tale-shape").style("stroke", "none");

      taleGroups.style("opacity", 1);
      g.selectAll(".stack-line").attr("opacity", 0.2);

      tooltip.style("opacity", 0);
    })
    .on("click", (event, d) => {
      console.log("Clicked:", d.tale);
    });

  // Draw legend
  drawLegend();

  // Initialize search functionality
  initSearch();

  console.log("Visualization complete!");
}

// Search functionality
function initSearch() {
  const searchInput = d3.select("#tale-search");
  const clearButton = d3.select("#clear-search");
  const resultsDiv = d3.select("#search-results");

  searchInput.on("input", function () {
    const query = this.value.toLowerCase().trim();

    if (query.length === 0) {
      resetSearch();
      resultsDiv.text("");
      return;
    }

    const matches = talesWithPositions.filter((d) =>
      d.tale.toLowerCase().includes(query)
    );

    if (matches.length === 0) {
      resultsDiv.text("No tales found").style("color", "#ff6b6b");
      dimAllTales();
    } else {
      const matchNames = matches
        .slice(0, 3)
        .map((d) => d.tale)
        .join(", ");
      const moreText =
        matches.length > 3 ? ` and ${matches.length - 3} more` : "";
      resultsDiv
        .text(`Found ${matches.length} tale(s): ${matchNames}${moreText}`)
        .style("color", "#51cf66");
      highlightMatches(matches);
    }
  });

  clearButton.on("click", () => {
    searchInput.node().value = "";
    resetSearch();
    resultsDiv.text("");
  });

  function highlightMatches(matches) {
    const matchTitles = new Set(matches.map((d) => d.tale));

    taleGroups
      .transition()
      .duration(300)
      .style("opacity", (d) => (matchTitles.has(d.tale) ? 1 : 0.15));

    // Also dim the connecting lines
    g.selectAll(".stack-line")
      .transition()
      .duration(300)
      .attr("opacity", (d) => (matchTitles.has(d.tale) ? 0.4 : 0.05));
  }

  function dimAllTales() {
    taleGroups.transition().duration(300).style("opacity", 0.15);

    g.selectAll(".stack-line").transition().duration(300).attr("opacity", 0.05);
  }

  function resetSearch() {
    taleGroups.transition().duration(300).style("opacity", 1);

    g.selectAll(".stack-line").transition().duration(300).attr("opacity", 0.2);
  }
}

// Biographical context
// function addBiographicalContext(g, timeSegments, timelineY, width, height) {
//   // Period backgrounds for the productive period only
//   const periods = [
//     {
//       start: timeSegments[2].displayStart,
//       end: timeSegments[2].displayEnd,
//       label: "Fairy Tale Period",
//       color: "rgba(248, 231, 28, 0.05)",
//     },
//   ];

//   periods.forEach((period) => {
//     g.append("rect")
//       .attr("x", period.start)
//       .attr("y", 0)
//       .attr("width", period.end - period.start)
//       .attr("height", height)
//       .attr("fill", period.color)
//       .attr("pointer-events", "none");

//     // Period label at top
//     g.append("text")
//       .attr("x", (period.start + period.end) / 2)
//       .attr("y", -15)
//       .attr("text-anchor", "middle")
//       .attr("fill", "#666")
//       .attr("font-size", "11px")
//       .attr("font-style", "italic")
//       .text(period.label);
//   });
// }

function calculateOpacity(purity) {
  if (isNaN(purity) || purity === null || purity === undefined) {
    return 0.5;
  }
  return Math.max(0.3, Math.min(1.0, 0.3 + purity * 1.4));
}

function showTooltip(event, d, tooltip) {
  const emotions = [
    { name: "Sadness", value: d.sadness, color: emotionColors.sadness },
    { name: "Joy", value: d.joy, color: emotionColors.joy },
    { name: "Fear", value: d.fear, color: emotionColors.fear },
    { name: "Anger", value: d.anger, color: emotionColors.anger },
    { name: "Disgust", value: d.disgust, color: emotionColors.disgust },
    { name: "Surprise", value: d.surprise, color: emotionColors.surprise },
    { name: "Neutral", value: d.neutral, color: emotionColors.neutral },
  ].sort((a, b) => b.value - a.value);

  const emotionBars = emotions
    .map(
      (e) => `
    <div class="tooltip-emotion-bar">
      <div class="tooltip-emotion-label">${e.name}</div>
      <div class="tooltip-emotion-value">
        <div class="tooltip-emotion-fill" style="width: ${
          e.value * 100
        }%; background-color: ${e.color};"></div>
      </div>
      <div class="tooltip-emotion-percent">${(e.value * 100).toFixed(0)}%</div>
    </div>
  `
    )
    .join("");

  const content = `
    <div class="tooltip-title">${d.tale}</div>
    <div class="tooltip-content">
      <div style="margin-bottom: 8px;">
        <strong>Published:</strong> ${d.year}${
    d.approximate ? " (approximate)" : ""
  }<br>
        <strong>Length:</strong> ${d.num_chunks} chunks<br>
        <strong>Date Certainty:</strong> ${
          d.approximate ? "Approximate" : "Confirmed"
        }
      </div>
      <div style="margin-bottom: 4px;"><strong>Emotional Profile:</strong></div>
      ${emotionBars}
    </div>
  `;

  tooltip
    .html(content)
    .style("left", event.pageX + 15 + "px")
    .style("top", event.pageY - 15 + "px")
    .style("opacity", 1);
}

function drawLegend() {
  const legend = d3.select("#legend");
  legend.html(""); // Clear existing

  // Emotion colors
  Object.entries(emotionColors).forEach(([emotion, color]) => {
    const item = legend.append("div").attr("class", "legend-item");

    item
      .append("div")
      .attr("class", "legend-circle")
      .style("background-color", color);

    item
      .append("span")
      .attr("class", "legend-label")
      .text(emotion.charAt(0).toUpperCase() + emotion.slice(1));
  });

  // Shape legend
  const shapeItem = legend
    .append("div")
    .attr("class", "legend-item")
    .style("margin-left", "20px");

  shapeItem
    .append("div")
    .style("width", "16px")
    .style("height", "16px")
    .style("background-color", "#666")
    .style("transform", "rotate(45deg)");

  shapeItem
    .append("span")
    .attr("class", "legend-label")
    .text("Approximate date");

  // Stacking note
  const noteItem = legend
    .append("div")
    .attr("class", "legend-item")
    .style("margin-left", "20px")
    .style("font-style", "italic")
    .style("color", "#888");

  noteItem.append("span").text("Tales stack by: certainty → emotion → size");
}

init();
