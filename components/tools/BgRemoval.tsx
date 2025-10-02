
import React, { useState, useCallback } from 'react';
import { editImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface BgRemovalProps {
    tool: Tool;
    onBack: () => void;
}

const BgRemoval: React.FC<BgRemovalProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [resultImage, setResultImage] = useState<string | null>(null);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setImageFile(file);
            setResultImage(null);
            setImagePreview(URL.createObjectURL(file));
        }
    };

    const handleGenerate = useCallback(async () => {
        if (!imageFile) {
            setError('Please upload an image.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResultImage(null);
        try {
            const prompt = "Remove the background from this image, leaving only the main subject. Make the background transparent.";
            const response = await editImage(prompt, imageFile);
            setResultImage(response.image);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [imageFile]);

    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>
            
            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Sube una imagen y la IA eliminará el fondo automáticamente, dejando solo el sujeto principal con un fondo transparente, listo para descargar.
                </p>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg text-center">
                <div className="w-full p-4 border-2 border-dashed border-gray-600 rounded-lg">
                    <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="image-upload" disabled={isLoading} />
                    <label htmlFor="image-upload" className={`cursor-pointer ${isLoading ? 'cursor-not-allowed' : ''}`}>
                        {imagePreview ? (
                            <p>Imagen seleccionada. ¡Lista para procesar!</p>
                        ) : (
                            <p>Click to upload an image</p>
                        )}
                    </label>
                </div>
                 <button
                    onClick={handleGenerate}
                    disabled={isLoading || !imageFile}
                    className="mt-4 w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Eliminando Fondo...' : 'Eliminar Fondo'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                <div>
                    <h3 className="text-xl font-semibold mb-4 text-center">Original</h3>
                    {imagePreview && <img src={imagePreview} alt="Original" className="w-full aspect-square object-contain rounded-lg shadow-lg bg-gray-700 p-2" />}
                </div>
                 <div>
                    <h3 className="text-xl font-semibold mb-4 text-center">Resultado</h3>
                    <div className="w-full aspect-square object-contain rounded-lg shadow-lg bg-gray-700 p-2" style={{backgroundImage: 'repeating-conic-gradient(#808080 0% 25%, transparent 0% 50%)', backgroundSize: '16px 16px'}}>
                        {isLoading && <div className="w-full h-full flex items-center justify-center animate-pulse bg-gray-800/50"><p>Procesando...</p></div>}
                        {resultImage && <img src={resultImage} alt="Background removed" className="w-full h-full object-contain" />}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BgRemoval;
