import React, { useState, useCallback } from 'react';
import { editImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface TaskAutomationProps {
    tool: Tool;
    onBack: () => void;
}

interface Task {
    id: string;
    label: string;
    prompt: string;
}

interface Result {
    taskId: string;
    label: string;
    image: string;
}

const TASKS: Task[] = [
    { id: 'ig_post', label: 'Redimensionar para Post de Instagram (1:1)', prompt: 'Recreate this image with a 1:1 aspect ratio, ensuring the main subject is centered and fully visible. Pad if necessary, do not crop important details.' },
    { id: 'ig_story', label: 'Redimensionar para Story (9:16)', prompt: 'Recreate this image with a 9:16 aspect ratio for a story format. The main subject should be centered. Fill extra space with a complementary, blurred background derived from the image itself.' },
    { id: 'grayscale', label: 'Convertir a Blanco y Negro', prompt: 'Convert this image to a high-contrast black and white version.' },
    { id: 'vintage', label: 'Aplicar Estilo Vintage', prompt: 'Apply a vintage photo effect to this image, with slightly faded colors and a subtle grain texture.' },
    { id: 'vibrant', label: 'Aumentar Vibración de Colores', prompt: 'Enhance the colors of this image to make them more vibrant and saturated, as if for a commercial ad.' },
];

const TaskAutomation: React.FC<TaskAutomationProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set());
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [results, setResults] = useState<Result[]>([]);
    const [progress, setProgress] = useState('');

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setImageFile(file);
            setImagePreview(URL.createObjectURL(file));
            setResults([]);
            setSelectedTasks(new Set());
        }
    };

    const handleTaskToggle = (taskId: string) => {
        const newSelection = new Set(selectedTasks);
        if (newSelection.has(taskId)) {
            newSelection.delete(taskId);
        } else {
            newSelection.add(taskId);
        }
        setSelectedTasks(newSelection);
    };

    const handleRunAutomation = useCallback(async () => {
        if (!imageFile || selectedTasks.size === 0) {
            setError('Por favor, sube una imagen y selecciona al menos una tarea.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResults([]);
        
        const tasksToRun = TASKS.filter(task => selectedTasks.has(task.id));
        const newResults: Result[] = [];

        for (let i = 0; i < tasksToRun.length; i++) {
            const task = tasksToRun[i];
            setProgress(`Procesando tarea ${i + 1} de ${tasksToRun.length}: ${task.label}`);
            try {
                const response = await editImage(task.prompt, imageFile);
                if (response.image) {
                    newResults.push({ taskId: task.id, label: task.label, image: response.image });
                }
            } catch (err) {
                console.error(`Error en la tarea '${task.label}':`, err);
                // Opcional: añadir un resultado de error a la lista
            }
        }
        
        setResults(newResults);
        setProgress('');
        setIsLoading(false);
    }, [imageFile, selectedTasks]);
    
    const downloadImage = (imageUrl: string, label: string) => {
        const link = document.createElement('a');
        link.href = imageUrl;
        link.download = `result_${label.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="max-w-6xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Columna de Configuración */}
                <div className="space-y-6">
                    <div>
                        <h3 className="text-xl font-semibold mb-2">1. Sube tu Imagen</h3>
                        <div className="w-full p-4 border-2 border-dashed border-gray-600 rounded-lg text-center bg-gray-800/50">
                            <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="image-upload-automation" disabled={isLoading} />
                            <label htmlFor="image-upload-automation" className={`cursor-pointer ${isLoading ? 'cursor-not-allowed' : ''}`}>
                                {imagePreview ? (
                                    <img src={imagePreview} alt="Preview" className="max-h-48 mx-auto rounded-lg" />
                                ) : (
                                    <p className="py-12">Haz clic para subir una imagen</p>
                                )}
                            </label>
                        </div>
                    </div>
                    <div>
                        <h3 className="text-xl font-semibold mb-2">2. Elige las Tareas</h3>
                        <div className="space-y-2">
                            {TASKS.map(task => (
                                <div key={task.id} className="flex items-center p-3 bg-gray-800 rounded-md">
                                    <input
                                        type="checkbox"
                                        id={task.id}
                                        checked={selectedTasks.has(task.id)}
                                        onChange={() => handleTaskToggle(task.id)}
                                        className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-purple-600 focus:ring-purple-500"
                                        disabled={isLoading}
                                    />
                                    <label htmlFor={task.id} className="ml-3 text-sm text-gray-300">{task.label}</label>
                                </div>
                            ))}
                        </div>
                    </div>
                    <button
                        onClick={handleRunAutomation}
                        disabled={isLoading || !imageFile || selectedTasks.size === 0}
                        className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                    >
                        {isLoading ? <Spinner /> : `Ejecutar ${selectedTasks.size} Tarea(s)`}
                    </button>
                    {error && <div className="mt-4 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
                </div>
                
                {/* Columna de Resultados */}
                <div>
                    <h3 className="text-xl font-semibold mb-2">Resultados</h3>
                    <div className="bg-gray-800/50 p-4 rounded-lg min-h-[300px]">
                        {isLoading && (
                             <div className="flex flex-col items-center justify-center h-full">
                                <Spinner />
                                <p className="mt-4 text-gray-400">{progress || 'Iniciando automatización...'}</p>
                            </div>
                        )}
                        {!isLoading && results.length > 0 && (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                {results.map(result => (
                                    <div key={result.taskId} className="group relative bg-gray-900 rounded-lg p-2">
                                        <img src={result.image} alt={result.label} className="w-full aspect-auto object-contain rounded-md" />
                                        <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center rounded-lg p-2">
                                            <p className="text-white text-xs text-center font-semibold mb-2">{result.label}</p>
                                            <button onClick={() => downloadImage(result.image, result.label)} className="text-white bg-purple-600/80 hover:bg-purple-600 px-3 py-1.5 text-sm rounded-lg font-semibold">
                                                Descargar
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                         {!isLoading && results.length === 0 && (
                            <div className="flex items-center justify-center h-full">
                                <p className="text-gray-500">Los resultados aparecerán aquí.</p>
                            </div>
                         )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TaskAutomation;
