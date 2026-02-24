import { useEffect, useRef, useState } from 'react'

export default function CustomCursor() {
    const dotRef = useRef(null)
    const ringRef = useRef(null)
    const mouse = useRef({ x: 0, y: 0 })
    const ring = useRef({ x: 0, y: 0 })
    const [hovering, setHovering] = useState(false)
    const raf = useRef(null)

    useEffect(() => {
        // hide default cursor
        document.body.style.cursor = 'none'

        const onMove = (e) => {
            mouse.current.x = e.clientX
            mouse.current.y = e.clientY
            if (dotRef.current) {
                dotRef.current.style.left = e.clientX + 'px'
                dotRef.current.style.top = e.clientY + 'px'
            }
        }

        const onOver = (e) => {
            const tag = e.target.tagName.toLowerCase()
            const interactive = e.target.closest('button, a, input, textarea, select, [role="button"], .tab-btn, .preview-tab, .amenity-item, .faq-question')
            setHovering(!!interactive || tag === 'button' || tag === 'a')
        }

        const animate = () => {
            ring.current.x += (mouse.current.x - ring.current.x) * 0.15
            ring.current.y += (mouse.current.y - ring.current.y) * 0.15
            if (ringRef.current) {
                ringRef.current.style.left = ring.current.x + 'px'
                ringRef.current.style.top = ring.current.y + 'px'
            }
            raf.current = requestAnimationFrame(animate)
        }

        window.addEventListener('mousemove', onMove)
        window.addEventListener('mouseover', onOver)
        raf.current = requestAnimationFrame(animate)

        return () => {
            document.body.style.cursor = ''
            window.removeEventListener('mousemove', onMove)
            window.removeEventListener('mouseover', onOver)
            cancelAnimationFrame(raf.current)
        }
    }, [])

    // Don't render on touch devices
    if (typeof window !== 'undefined' && window.matchMedia('(hover: none)').matches) {
        return null
    }

    return (
        <>
            <div ref={dotRef} className={`cursor-dot${hovering ? ' hovering' : ''}`} />
            <div ref={ringRef} className={`cursor-ring${hovering ? ' hovering' : ''}`} />
        </>
    )
}
