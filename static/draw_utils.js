// static/draw_utils.js

/**
 * Main helper to draw a connection between two nodes.
 * Decides whether to draw a straight line or a curved bezier based on grid distance.
 */
function drawConnection(u, v, nodePositions, scenarioData, col, weight) {
    let p1 = nodePositions[u];
    let p2 = nodePositions[v];

    // Safety check: skip if nodes don't have positions
    if (!p1 || !p2) return;

    // Check grid distance to decide curve
    let n1 = scenarioData.nodes.find(n => n.id === u);
    let n2 = scenarioData.nodes.find(n => n.id === v);

    // Safety check: skip if node data is missing
    if (!n1 || !n2) return;

    // Heuristic: If nodes are not adjacent in the 3x3 grid, curve the edge
    let isLong = (abs(n1.grid_x - n2.grid_x) > 1 || abs(n1.grid_y - n2.grid_y) > 1);

    if (isLong) {
        drawCurvedEdge(p1, p2, col, weight);
    } else {
        drawStraightEdge(p1, p2, col, weight);
    }
}

/**
 * Draws a straight line with an arrowhead.
 */
function drawStraightEdge(p1, p2, col, weight) {
    stroke(col);
    strokeWeight(weight);
    noFill();

    line(p1.x, p1.y, p2.x, p2.y);

    drawArrowHead(p1.x, p1.y, p2.x, p2.y, col, weight);
}

/**
 * Draws a quadratic bezier curve with an arrowhead.
 * Calculates a control point offset perpendicular to the midpoint.
 */
function drawCurvedEdge(p1, p2, col, weight) {
    stroke(col);
    strokeWeight(weight);
    noFill();

    // Math for Control Point
    let mx = (p1.x + p2.x) / 2;
    let my = (p1.y + p2.y) / 2;
    let vx = p2.x - p1.x;
    let vy = p2.y - p1.y;

    // Curvature scale
    let scale = 0.25;
    let cx = mx - vy * scale;
    let cy = my + vx * scale;

    beginShape();
    vertex(p1.x, p1.y);
    quadraticVertex(cx, cy, p2.x, p2.y);
    endShape();

    drawArrowHead(cx, cy, p2.x, p2.y, col, weight);
}

/**
 * Draws a triangle arrow at the end of the line/curve.
 * Adjusts the stopping point based on line weight (highlight vs normal).
 */
function drawArrowHead(fromX, fromY, toX, toY, col, weight) {
    let angle = atan2(toY - fromY, toX - fromX);

    // If weight > 2 (highlighted), stop further back from the node center
    // so the thick arrow doesn't clip ugly into the circle
    let stopRadius = (weight > 2) ? 40 : 32;

    push();
    translate(toX, toY);
    rotate(angle);
    translate(-stopRadius, 0);

    fill(col);
    noStroke();
    triangle(0, 0, -10, -5, -10, 5);
    pop();
}