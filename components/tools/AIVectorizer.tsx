import React, { useState, useCallback, useRef, useEffect } from 'react';
import { editImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface AIVectorizerProps {
    tool: Tool;
    onBack: () => void;
}

const CameraView: React.FC<{ onCapture: (file: File) => void, onCancel: () => void }> = ({ onCapture, onCancel }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    useEffect(() => {
        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                streamRef.current = stream;
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (err) {
                console.error("Error accessing camera:", err);
                alert("Could not access camera. Please check permissions.");
                onCancel();
            }
        };
        startCamera();

        return () => {
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
        };
    }, [onCancel]);

    const handleCapture = () => {
        if (videoRef.current && canvasRef.current) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context?.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
            canvas.toBlob(blob => {
                if (blob) {
                    const file = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
                    onCapture(file);
                }
            }, 'image/jpeg');
        }
    };

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 p-4 rounded-lg w-full max-w-lg">
                <video ref={videoRef} autoPlay playsInline className="w-full rounded-md"></video>
                <canvas ref={canvasRef} className="hidden"></canvas>
                <div className="flex justify-center gap-4 mt-4">
                    <button onClick={handleCapture} className="px-6 py-2 bg-purple-600 rounded-md text-white font-semibold">Capturar</button>
                    <button onClick={onCancel} className="px-6 py-2 bg-gray-600 rounded-md text-white font-semibold">Cancelar</button>
                </div>
            </div>
        </div>
    );
};


const AIVectorizer: React.FC<AIVectorizerProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [resultImage, setResultImage] = useState<string | null>(null);
    const [isCameraOpen, setIsCameraOpen] = useState(false);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            processFile(file);
        }
    };
    
    const processFile = (file: File) => {
        setImageFile(file);
        setResultImage(null);
        setImagePreview(URL.createObjectURL(file));
        setIsCameraOpen(false);
    }

    const handleVectorize = useCallback(async () => {
        if (!imageFile) {
            setError('Please upload or capture an image.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResultImage(null);
        try {
            const prompt = "Analyze this image and create a clean, black and white line art vector trace of the main subject. The background should be completely white. The lines should be solid black and clearly defined.";
            const response = await editImage(prompt, imageFile);
            setResultImage(response.image);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [imageFile]);
    
    const handleExportPdf = () => {
        if (!resultImage) return;
        const printWindow = window.open('', '_blank');
        printWindow?.document.write(`
            <html>
                <head><title>Export</title></head>
                <body style="margin: 0; text-align: center;">
                    <img src="${resultImage}" style="max-width: 100%; height: auto;" />
                    <script>
                        setTimeout(() => { 
                            window.print();
                            window.close();
                        }, 500);
                    </script>
                </body>
            </html>
        `);
        printWindow?.document.close();
    };

    return (
        <div className="max-w-4xl mx-auto">
            {isCameraOpen && <CameraView onCapture={processFile} onCancel={() => setIsCameraOpen(false)} />}
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Sube una foto de un dibujo, boceto o cualquier imagen, o toma una foto con tu cámara. La IA lo convertirá en un trazo de líneas limpias. Luego, puedes exportar el resultado a PDF.
                </p>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg text-center">
                <div className="flex flex-col sm:flex-row gap-4">
                     <label htmlFor="image-upload" className="w-full cursor-pointer bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 px-4 rounded-lg transition-colors">
                        Subir Imagen
                     </label>
                     <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="image-upload" disabled={isLoading} />
                     <button onClick={() => setIsCameraOpen(true)} className="w-full bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 px-4 rounded-lg transition-colors">Tomar Foto</button>
                </div>
                 <button
                    onClick={handleVectorize}
                    disabled={isLoading || !imageFile}
                    className="mt-4 w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Vectorizando...' : 'Vectorizar Imagen'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                <div>
                    <h3 className="text-xl font-semibold mb-4 text-center">Original</h3>
                    <div className="w-full aspect-square bg-gray-700 rounded-lg p-2 flex items-center justify-center">
                       {imagePreview ? <img src={imagePreview} alt="Original" className="w-full h-full object-contain rounded-md" /> : <p className="text-gray-400">Sube o captura una imagen</p>}
                    </div>
                </div>
                 <div>
                    <h3 className="text-xl font-semibold mb-4 text-center">Vectorizado</h3>
                     <div className="w-full aspect-square bg-white rounded-lg shadow-lg p-2">
                        {isLoading && <div className="w-full h-full flex items-center justify-center animate-pulse bg-gray-800/50 rounded-md"><p className="text-white">Procesando...</p></div>}
                        {resultImage && <img src={resultImage} alt="Vectorized result" className="w-full h-full object-contain" />}
                    </div>
                </div>
            </div>
            
             {resultImage && !isLoading && (
                <div className="mt-8 text-center">
                    <button onClick={handleExportPdf} className="bg-green-600 hover:bg-green-500 text-white font-bold py-3 px-8 rounded-lg transition-colors">
                        Exportar a PDF
                    </button>
                </div>
            )}
        </div>
    );
};

export default AIVectorizer;