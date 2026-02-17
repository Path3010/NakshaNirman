import { Suspense, useState, useEffect } from 'react'
import { Canvas, useLoader } from '@react-three/fiber'
import { OrbitControls, Center } from '@react-three/drei'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

function Model({ url }) {
    const gltf = useLoader(GLTFLoader, url)
    return <primitive object={gltf.scene} />
}

export default function Viewer3D({ projectId }) {
    const [modelUrl, setModelUrl] = useState(null)
    const [generating, setGenerating] = useState(false)
    const [error, setError] = useState(null)

    useEffect(() => {
        if (!projectId) return

        let cancelled = false

        const generateAndLoad = async () => {
            setGenerating(true)
            setError(null)
            setModelUrl(null)

            try {
                // Step 1: Generate the 3D model
                const res = await fetch(`/api/generate-3d/${projectId}`, { method: 'POST' })

                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}))
                    throw new Error(errData.detail || `Failed to generate 3D model (${res.status})`)
                }

                const data = await res.json()
                if (!cancelled) {
                    // Step 2: Set the model URL so the Canvas loads it
                    setModelUrl(data.model_url || `/api/3d-model/${projectId}`)
                }
            } catch (err) {
                if (!cancelled) {
                    setError(err.message || 'Failed to generate 3D model')
                }
            } finally {
                if (!cancelled) {
                    setGenerating(false)
                }
            }
        }

        generateAndLoad()

        return () => { cancelled = true }
    }, [projectId])

    if (generating) {
        return (
            <div className="viewer-3d" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '0.75rem' }}>
                <div className="spinner"></div>
                <span className="loading-text">Generating 3D model...</span>
            </div>
        )
    }

    if (error) {
        return (
            <div className="viewer-3d" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '0.75rem', padding: '2rem' }}>
                <svg width="40" height="40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.2" style={{ color: 'var(--error)' }}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>3D Generation Failed</span>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', textAlign: 'center', maxWidth: 320 }}>{error}</span>
                <button className="btn btn-primary btn-sm" onClick={() => {
                    setGenerating(true)
                    setError(null)
                    fetch(`/api/generate-3d/${projectId}`, { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                            setModelUrl(data.model_url || `/api/3d-model/${projectId}`)
                            setGenerating(false)
                        })
                        .catch(err => {
                            setError(err.message || 'Retry failed')
                            setGenerating(false)
                        })
                }} style={{ marginTop: '0.5rem' }}>
                    Retry
                </button>
            </div>
        )
    }

    if (!modelUrl) {
        return (
            <div className="viewer-3d" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span className="loading-text">Waiting for 3D model...</span>
            </div>
        )
    }

    return (
        <div className="viewer-3d">
            <Canvas
                camera={{ position: [50, 50, 50], fov: 50 }}
                style={{ background: '#f5f5f5' }}
            >
                <ambientLight intensity={0.6} />
                <directionalLight position={[30, 50, 30]} intensity={1} castShadow />
                <pointLight position={[-20, 30, -20]} intensity={0.4} color="#FF6500" />

                <Suspense fallback={null}>
                    <Center>
                        <Model url={modelUrl} />
                    </Center>
                </Suspense>

                <OrbitControls
                    enableDamping
                    dampingFactor={0.05}
                    minDistance={10}
                    maxDistance={200}
                />

                <gridHelper args={[200, 50, '#ccc', '#e0e0e0']} position={[0, -0.5, 0]} />
            </Canvas>
        </div>
    )
}
