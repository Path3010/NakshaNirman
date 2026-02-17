import { useMemo } from 'react'

/* ──────────────────────────────────────────────
   Furniture SVG helpers  (all return JSX groups)
   Drawn relative to room centroid and scaled
   ────────────────────────────────────────────── */
function furnitureBed(cx, cy, w, h, scale) {
    const bw = Math.min(w * 0.55, h * 0.7) * scale
    const bh = bw * 0.6
    return (
        <g>
            {/* Bed frame */}
            <rect x={cx - bw / 2} y={cy - bh / 2} width={bw} height={bh}
                fill="none" stroke="#333" strokeWidth={scale * 0.4} rx={scale * 0.3} />
            {/* Pillows */}
            <rect x={cx - bw / 2 + bw * 0.08} y={cy - bh / 2 + bh * 0.08}
                width={bw * 0.35} height={bh * 0.25}
                fill="none" stroke="#555" strokeWidth={scale * 0.25} rx={scale * 0.2} />
            <rect x={cx + bw / 2 - bw * 0.43} y={cy - bh / 2 + bh * 0.08}
                width={bw * 0.35} height={bh * 0.25}
                fill="none" stroke="#555" strokeWidth={scale * 0.25} rx={scale * 0.2} />
        </g>
    )
}

function furnitureSofa(cx, cy, w, h, scale) {
    const sw = Math.min(w * 0.5, h * 0.6) * scale
    const sh = sw * 0.4
    return (
        <g>
            {/* Sofa base */}
            <rect x={cx - sw / 2} y={cy - sh / 2} width={sw} height={sh}
                fill="none" stroke="#333" strokeWidth={scale * 0.4} rx={scale * 0.5} />
            {/* Back rest */}
            <rect x={cx - sw / 2} y={cy - sh / 2} width={sw} height={sh * 0.3}
                fill="none" stroke="#555" strokeWidth={scale * 0.3} rx={scale * 0.4} />
            {/* Coffee table */}
            <ellipse cx={cx} cy={cy + sh * 0.7} rx={sw * 0.25} ry={sw * 0.12}
                fill="none" stroke="#777" strokeWidth={scale * 0.25} />
        </g>
    )
}

function furnitureKitchen(cx, cy, w, h, scale) {
    const kw = Math.min(w * 0.45, h * 0.65) * scale
    const kh = kw * 0.35
    return (
        <g>
            {/* Counter */}
            <rect x={cx - kw / 2} y={cy - kh / 2} width={kw} height={kh}
                fill="none" stroke="#333" strokeWidth={scale * 0.4} />
            {/* Sink circle */}
            <circle cx={cx - kw * 0.15} cy={cy} r={kw * 0.08}
                fill="none" stroke="#555" strokeWidth={scale * 0.25} />
            {/* Stove circles */}
            <circle cx={cx + kw * 0.15} cy={cy - kh * 0.15} r={kw * 0.06}
                fill="none" stroke="#555" strokeWidth={scale * 0.25} />
            <circle cx={cx + kw * 0.3} cy={cy - kh * 0.15} r={kw * 0.06}
                fill="none" stroke="#555" strokeWidth={scale * 0.25} />
        </g>
    )
}

function furnitureBathroom(cx, cy, w, h, scale) {
    const bs = Math.min(w, h) * 0.35 * scale
    return (
        <g>
            {/* Toilet */}
            <ellipse cx={cx} cy={cy + bs * 0.3} rx={bs * 0.3} ry={bs * 0.4}
                fill="none" stroke="#333" strokeWidth={scale * 0.35} />
            <rect x={cx - bs * 0.25} y={cy - bs * 0.3} width={bs * 0.5} height={bs * 0.35}
                fill="none" stroke="#555" strokeWidth={scale * 0.25} rx={scale * 0.3} />
        </g>
    )
}

function furnitureDining(cx, cy, w, h, scale) {
    const ts = Math.min(w * 0.4, h * 0.5) * scale
    return (
        <g>
            {/* Table */}
            <rect x={cx - ts / 2} y={cy - ts * 0.35} width={ts} height={ts * 0.7}
                fill="none" stroke="#333" strokeWidth={scale * 0.4} rx={scale * 0.3} />
            {/* Chairs */}
            {[-1, 1].map(dx => [-1, 1].map(dy => (
                <rect key={`${dx}${dy}`}
                    x={cx + dx * (ts * 0.65) - ts * 0.1}
                    y={cy + dy * (ts * 0.15) - ts * 0.1}
                    width={ts * 0.2} height={ts * 0.2}
                    fill="none" stroke="#777" strokeWidth={scale * 0.2} rx={scale * 0.15} />
            )))}
        </g>
    )
}

function furnitureStudy(cx, cy, w, h, scale) {
    const ds = Math.min(w * 0.45, h * 0.5) * scale
    return (
        <g>
            {/* Desk */}
            <rect x={cx - ds / 2} y={cy - ds * 0.25} width={ds} height={ds * 0.5}
                fill="none" stroke="#333" strokeWidth={scale * 0.4} />
            {/* Chair */}
            <circle cx={cx} cy={cy + ds * 0.5} r={ds * 0.15}
                fill="none" stroke="#777" strokeWidth={scale * 0.25} />
        </g>
    )
}

const FURNITURE_MAP = {
    master_bedroom: furnitureBed,
    bedroom: furnitureBed,
    living: furnitureSofa,
    kitchen: furnitureKitchen,
    bathroom: furnitureBathroom,
    dining: furnitureDining,
    study: furnitureStudy,
}

/* ──────────────── Main Component ──────────────── */

export default function PlanPreview({ plan }) {
    const layout = useMemo(() => {
        if (!plan || !plan.boundary || plan.boundary.length < 3) return null

        const allPoints = [
            ...plan.boundary,
            ...plan.rooms.flatMap(r => r.polygon || []),
        ]
        const xs = allPoints.map(p => p[0])
        const ys = allPoints.map(p => p[1])
        const minX = Math.min(...xs)
        const minY = Math.min(...ys)
        const maxX = Math.max(...xs)
        const maxY = Math.max(...ys)
        const w = maxX - minX
        const h = maxY - minY
        const pad = Math.max(w, h) * 0.15
        const viewBox = `${minX - pad} ${minY - pad} ${w + pad * 2} ${h + pad * 2}`
        const scale = Math.max(w, h)

        return { viewBox, minX, minY, maxX, maxY, w, h, scale, pad }
    }, [plan])

    if (!layout) return <div className="preview-empty"><p>No plan data</p></div>

    const { viewBox, minX, minY, w, h, scale } = layout

    const toPathD = (coords) => {
        if (!coords || coords.length < 2) return ''
        return coords.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ') + ' Z'
    }

    const wallThick = scale * 0.006
    const innerWall = scale * 0.003

    // Compute room dimensions for label
    const getRoomDims = (polygon) => {
        if (!polygon || polygon.length < 3) return null
        const rxs = polygon.map(p => p[0])
        const rys = polygon.map(p => p[1])
        const rw = Math.max(...rxs) - Math.min(...rxs)
        const rh = Math.max(...rys) - Math.min(...rys)
        return { w: rw, h: rh }
    }

    return (
        <div className="plan-svg" style={{ padding: '1rem', background: '#fff' }}>
            <svg viewBox={viewBox} xmlns="http://www.w3.org/2000/svg" style={{ background: '#fff' }}>
                <defs>
                    {/* Wall pattern for outer walls */}
                    <filter id="wall-shadow" x="-5%" y="-5%" width="110%" height="110%">
                        <feDropShadow dx="0" dy="0" stdDeviation={scale * 0.002} floodColor="#000" floodOpacity="0.1" />
                    </filter>
                </defs>

                {/* Background */}
                <rect
                    x={minX - layout.pad}
                    y={minY - layout.pad}
                    width={w + layout.pad * 2}
                    height={h + layout.pad * 2}
                    fill="#ffffff"
                />

                {/* Outer boundary — thick black walls */}
                <path
                    d={toPathD(plan.boundary)}
                    fill="#fafafa"
                    stroke="#1a1a1a"
                    strokeWidth={wallThick}
                    strokeLinejoin="miter"
                    filter="url(#wall-shadow)"
                />

                {/* Room polygons — white fill, thin black interior walls */}
                {plan.rooms.map((room, i) => (
                    <path
                        key={`room-${i}`}
                        d={toPathD(room.polygon)}
                        fill="#ffffff"
                        stroke="#1a1a1a"
                        strokeWidth={innerWall}
                        strokeLinejoin="miter"
                    />
                ))}

                {/* Furniture icons */}
                {plan.rooms.map((room, i) => {
                    const fn = FURNITURE_MAP[room.room_type]
                    if (!fn || !room.centroid || !room.polygon) return null
                    const dims = getRoomDims(room.polygon)
                    if (!dims) return null
                    return <g key={`furn-${i}`}>{fn(room.centroid[0], room.centroid[1], dims.w, dims.h, 1)}</g>
                })}

                {/* Door arcs */}
                {plan.doors?.map((door, i) => {
                    if (!door.hinge) {
                        // fallback: small circle
                        return (
                            <circle key={`door-${i}`}
                                cx={door.position[0]} cy={door.position[1]}
                                r={scale * 0.005} fill="none" stroke="#333" strokeWidth={scale * 0.0015}
                            />
                        )
                    }
                    const r = door.width
                    const hx = door.hinge[0]
                    const hy = door.hinge[1]
                    const sx = door.swing_dir[0]
                    const sy = door.swing_dir[1]
                    // Arc end point
                    const arcEndX = hx + sx * r
                    const arcEndY = hy + sy * r
                    // Door line (the panel)
                    const doorEndX = door.door_end[0]
                    const doorEndY = door.door_end[1]

                    return (
                        <g key={`door-${i}`}>
                            {/* Wall gap (white line to clear the wall) */}
                            <line
                                x1={hx} y1={hy} x2={doorEndX} y2={doorEndY}
                                stroke="#fff" strokeWidth={wallThick * 1.2}
                            />
                            {/* Door panel */}
                            <line
                                x1={hx} y1={hy} x2={arcEndX} y2={arcEndY}
                                stroke="#333" strokeWidth={scale * 0.0015}
                            />
                            {/* Arc sweep (quarter circle) */}
                            <path
                                d={`M ${arcEndX},${arcEndY} A ${r},${r} 0 0 ${sy > 0 || sx < 0 ? 1 : 0} ${doorEndX},${doorEndY}`}
                                fill="none" stroke="#333"
                                strokeWidth={scale * 0.0012}
                                strokeDasharray={`${scale * 0.003} ${scale * 0.003}`}
                            />
                        </g>
                    )
                })}

                {/* Windows — three parallel lines */}
                {plan.windows?.map((win, i) => {
                    if (!win.start || !win.end) {
                        return (
                            <rect key={`win-${i}`}
                                x={win.position[0] - scale * 0.01}
                                y={win.position[1] - scale * 0.002}
                                width={scale * 0.02} height={scale * 0.004}
                                fill="#fff" stroke="#333" strokeWidth={scale * 0.001}
                            />
                        )
                    }

                    const sx = win.start[0], sy = win.start[1]
                    const ex = win.end[0], ey = win.end[1]
                    const dx = ex - sx, dy = ey - sy
                    const len = Math.sqrt(dx * dx + dy * dy)
                    if (len < 0.1) return null
                    const nx = -dy / len, ny = dx / len
                    const offset = scale * 0.003

                    return (
                        <g key={`win-${i}`}>
                            {/* Clear the wall */}
                            <line x1={sx} y1={sy} x2={ex} y2={ey}
                                stroke="#fff" strokeWidth={wallThick * 1.3} />
                            {/* Three parallel lines */}
                            {[-1, 0, 1].map(m => (
                                <line key={m}
                                    x1={sx + nx * offset * m} y1={sy + ny * offset * m}
                                    x2={ex + nx * offset * m} y2={ey + ny * offset * m}
                                    stroke="#1a1a1a" strokeWidth={scale * 0.0012}
                                />
                            ))}
                        </g>
                    )
                })}

                {/* Room labels */}
                {plan.rooms.map((room, i) => {
                    if (!room.centroid) return null
                    const cx = room.centroid[0]
                    const cy = room.centroid[1]
                    const dims = getRoomDims(room.polygon)
                    const furn = FURNITURE_MAP[room.room_type]
                    // Push label down if furniture is drawn
                    const labelOffset = furn && dims ? Math.min(dims.w, dims.h) * 0.3 : 0

                    return (
                        <g key={`label-${i}`}>
                            <text
                                x={cx} y={cy - scale * 0.01 + labelOffset}
                                textAnchor="middle"
                                fill="#1a1a1a"
                                fontSize={scale * 0.018}
                                fontWeight="700"
                                fontFamily="Inter, sans-serif"
                                letterSpacing={scale * 0.002}
                            >
                                {room.label?.toUpperCase()}
                            </text>
                            <text
                                x={cx} y={cy + scale * 0.015 + labelOffset}
                                textAnchor="middle"
                                fill="#888"
                                fontSize={scale * 0.013}
                                fontFamily="Inter, sans-serif"
                            >
                                {room.actual_area?.toFixed(0)} sq ft
                            </text>
                        </g>
                    )
                })}

                {/* Dimension lines along boundary edges */}
                {plan.boundary && plan.boundary.length > 2 && (() => {
                    const pts = plan.boundary
                    const dimLines = []
                    for (let i = 0; i < pts.length - 1; i++) {
                        const [x1, y1] = pts[i]
                        const [x2, y2] = pts[i + 1]
                        const dx = x2 - x1, dy = y2 - y1
                        const segLen = Math.sqrt(dx * dx + dy * dy)
                        if (segLen < 3) continue

                        const mx = (x1 + x2) / 2
                        const my = (y1 + y2) / 2
                        const nx = -dy / segLen, ny = dx / segLen
                        const dimOffset = scale * 0.035

                        // Offset the dimension line outward
                        const ox1 = x1 + nx * dimOffset
                        const oy1 = y1 + ny * dimOffset
                        const ox2 = x2 + nx * dimOffset
                        const oy2 = y2 + ny * dimOffset
                        const omx = mx + nx * dimOffset
                        const omy = my + ny * dimOffset

                        // Determine rotation for text
                        let angle = Math.atan2(dy, dx) * 180 / Math.PI
                        if (angle > 90 || angle < -90) angle += 180

                        dimLines.push(
                            <g key={`dim-${i}`}>
                                {/* Extension lines */}
                                <line x1={x1} y1={y1}
                                    x2={x1 + nx * (dimOffset + scale * 0.005)}
                                    y2={y1 + ny * (dimOffset + scale * 0.005)}
                                    stroke="#999" strokeWidth={scale * 0.001} />
                                <line x1={x2} y1={y2}
                                    x2={x2 + nx * (dimOffset + scale * 0.005)}
                                    y2={y2 + ny * (dimOffset + scale * 0.005)}
                                    stroke="#999" strokeWidth={scale * 0.001} />
                                {/* Dimension line */}
                                <line x1={ox1} y1={oy1} x2={ox2} y2={oy2}
                                    stroke="#999" strokeWidth={scale * 0.001} />
                                {/* Arrows */}
                                <circle cx={ox1} cy={oy1} r={scale * 0.002} fill="#999" />
                                <circle cx={ox2} cy={oy2} r={scale * 0.002} fill="#999" />
                                {/* Text */}
                                <text
                                    x={omx} y={omy - scale * 0.005}
                                    textAnchor="middle"
                                    fill="#666"
                                    fontSize={scale * 0.012}
                                    fontFamily="Inter, sans-serif"
                                    transform={`rotate(${angle}, ${omx}, ${omy - scale * 0.005})`}
                                >
                                    {segLen.toFixed(1)} ft
                                </text>
                            </g>
                        )
                    }
                    return dimLines
                })()}

                {/* Title block */}
                <text
                    x={minX}
                    y={minY - h * 0.06}
                    fill="#1a1a1a"
                    fontSize={scale * 0.028}
                    fontWeight="800"
                    fontFamily="Inter, sans-serif"
                    letterSpacing={scale * 0.004}
                >
                    FLOOR PLAN
                </text>
                <text
                    x={minX}
                    y={minY - h * 0.025}
                    fill="#888"
                    fontSize={scale * 0.015}
                    fontFamily="Inter, sans-serif"
                >
                    Total: {plan.total_area?.toFixed(0)} sq ft  |  {plan.rooms.length} rooms
                </text>
            </svg>
        </div>
    )
}
