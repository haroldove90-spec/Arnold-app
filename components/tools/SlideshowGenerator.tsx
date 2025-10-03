import React, { useState, useCallback } from 'react';
import { generateSlideshow } from '../../services/geminiService';
import type { Tool } from '../../types';
import type { SlideshowData, Slide } from '../../services/geminiService';
import Spinner from '../ui/Spinner';

interface SlideshowGeneratorProps {
    tool: Tool;
    onBack: () => void;
}

const SlideshowGenerator: React.FC<SlideshowGeneratorProps> = ({ tool, onBack }) => {
    const [files, setFiles] = useState<File[]>([]);
    const [previews, setPreviews] = useState<string[]>([]);
    const [title, setTitle] = useState('');
    const [points, setPoints] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [slideshow, setSlideshow] = useState<SlideshowData | null>(null);
    const [currentSlide, setCurrentSlide] = useState(0);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        // FIX: Explicitly type selectedFiles as File[] to ensure correct type inference for URL.createObjectURL.
        const selectedFiles: File[] = Array.from(e.target.files || []);
        if (selectedFiles.length > 5) {
            alert("Puedes subir un máximo de 5 imágenes.");
            return;
        }
        setFiles(selectedFiles);
        const newPreviews = selectedFiles.map(file => URL.createObjectURL(file));
        setPreviews(newPreviews);
    };

    const handleGenerate = useCallback(async () => {
        if (files.length === 0) {
            setError('Por favor, sube al menos una imagen.');
            return;
        }
        if (!title.trim() || !points.trim()) {
            setError('Por favor, completa el título y los puntos clave.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setSlideshow(null);
        setCurrentSlide(0);
        try {
            const data = await generateSlideshow(files, title, points);
            setSlideshow(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Ocurrió un error desconocido.');
        } finally {
            setIsLoading(false);
        }
    }, [files, title, points]);

    const renderSlide = (slide: Slide, index: number) => {
        const imageSrc = typeof slide.image_index === 'number' && previews[slide.image_index] 
            ? previews[slide.image_index] 
            : 'https://via.placeholder.com/800x600/1F2937/FFFFFF?text=Imagen+No+Encontrada';

        switch (slide.type) {
            case 'title':
                return (
                    <div className="flex flex-col items-center justify-center h-full text-center p-8">
                        <h2 className="text-5xl font-bold text-white">{slide.title || 'Título'}</h2>
                        <p className="text-2xl text-gray-400 mt-4">{slide.subtitle || 'Subtítulo'}</p>
                    </div>
                );
            case 'image_left':
                return (
                    <div className="flex flex-col md:flex-row h-full">
                        <div className="w-full md:w-1/2 h-1/2 md:h-full bg-black"><img src={imageSrc} className="w-full h-full object-cover" alt={slide.title}/></div>
                        <div className="w-full md:w-1/2 flex flex-col justify-center p-8">
                            <h3 className="text-3xl font-semibold text-purple-400">{slide.title}</h3>
                            <p className="mt-4 text-gray-300">{slide.text}</p>
                        </div>
                    </div>
                );
            case 'image_right':
                 return (
                    <div className="flex flex-col-reverse md:flex-row h-full">
                        <div className="w-full md:w-1/2 flex flex-col justify-center p-8">
                            <h3 className="text-3xl font-semibold text-purple-400">{slide.title}</h3>
                            <p className="mt-4 text-gray-300">{slide.text}</p>
                        </div>
                        <div className="w-full md:w-1/2 h-1/2 md:h-full bg-black"><img src={imageSrc} className="w-full h-full object-cover" alt={slide.title}/></div>
                    </div>
                );
            case 'full_image':
                return (
                    <div className="relative h-full">
                        <img src={imageSrc} className="w-full h-full object-contain" alt={slide.caption}/>
                        {slide.caption && <p className="absolute bottom-4 left-4 bg-black/50 text-white p-2 rounded">{slide.caption}</p>}
                    </div>
                );
            case 'bullet_points':
                return (
                    <div className="flex flex-col justify-center h-full p-8">
                         <h3 className="text-4xl font-semibold text-purple-400 mb-6">{slide.title}</h3>
                         <ul className="space-y-4">
                            {slide.points?.map((point, i) => (
                                <li key={i} className="text-xl text-gray-300 flex items-start">
                                    <svg className="w-6 h-6 text-purple-400 mr-3 mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                    {point}
                                </li>
                            ))}
                         </ul>
                    </div>
                );
            case 'end':
                 return (
                    <div className="flex flex-col items-center justify-center h-full text-center">
                        <h2 className="text-5xl font-bold text-white">{slide.title || 'Gracias'}</h2>
                    </div>
                );
            default:
                return <div className="p-8">Tipo de diapositiva no reconocido.</div>;
        }
    };

    if (slideshow) {
        return (
            <div className="animate-fade-in">
                 <div className="aspect-video w-full max-w-5xl mx-auto bg-gray-800 rounded-lg shadow-2xl relative overflow-hidden">
                    {renderSlide(slideshow.slides[currentSlide], currentSlide)}
                 </div>
                 <div className="flex justify-between items-center max-w-5xl mx-auto mt-4">
                    <button onClick={() => setCurrentSlide(s => Math.max(0, s - 1))} disabled={currentSlide === 0} className="px-4 py-2 bg-gray-700 rounded disabled:opacity-50">&larr; Anterior</button>
                    <p className="text-gray-400">{currentSlide + 1} / {slideshow.slides.length}</p>
                    <button onClick={() => setCurrentSlide(s => Math.min(slideshow.slides.length - 1, s + 1))} disabled={currentSlide === slideshow.slides.length - 1} className="px-4 py-2 bg-gray-700 rounded disabled:opacity-50">Siguiente &rarr;</button>
                 </div>
                 <div className="text-center mt-6">
                    <button onClick={() => setSlideshow(null)} className="text-purple-400 hover:text-purple-300">Crear otra presentación</button>
                 </div>
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="space-y-6 bg-gray-800/50 p-6 rounded-lg">
                <div>
                    <label className="block text-lg font-semibold mb-2">1. Sube tus Imágenes (hasta 5)</label>
                    <div className="p-4 border-2 border-dashed border-gray-600 rounded-lg bg-gray-800">
                        <input type="file" multiple accept="image/*" onChange={handleFileChange} className="hidden" id="slideshow-upload" disabled={isLoading} />
                        <label htmlFor="slideshow-upload" className="cursor-pointer text-center block">
                             <p className="text-purple-400 font-semibold">Seleccionar archivos</p>
                             <p className="text-xs text-gray-500 mt-1">Arrastra y suelta o haz clic aquí</p>
                        </label>
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                        {previews.map((src, i) => <img key={i} src={src} className="w-20 h-20 object-cover rounded" alt={`preview ${i}`}/>)}
                    </div>
                </div>

                <div>
                    <label htmlFor="presentation-title" className="block text-lg font-semibold mb-2">2. Define el Contexto</label>
                    <input id="presentation-title" type="text" value={title} onChange={e => setTitle(e.target.value)} placeholder="Título de la Presentación" className="w-full p-3 bg-gray-800 border-gray-700 rounded mb-2" />
                    <textarea value={points} onChange={e => setPoints(e.target.value)} placeholder="Puntos Clave (uno por línea)..." className="w-full p-3 bg-gray-800 border-gray-700 rounded" rows={4}></textarea>
                </div>
                
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Presentación...' : '3. Generar Presentación'}
                </button>
            </div>
             {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
        </div>
    );
};

export default SlideshowGenerator;