
import React from 'react';
import type { Tool } from './types';

// Icons
const SparklesIcon = (
    <svg xmlns="http://www.w.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" /></svg>
);
const AdjustmentsIcon = (
    <svg xmlns="http://www.w.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 16v-2m0-8v-2m-6 8v2m0-16V4m0 8v2m12 8v2m0-16V4m0 8v2M6 12h12" /></svg>
);
const ScissorsIcon = (
    <svg xmlns="http://www.w.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.121 14.121L19 19m-7-7l7-7m-7 7l-2.879 2.879a1 1 0 01-1.414 0L9 14m-4 4v1a3 3 0 003 3h1a3 3 0 003-3v-1m-6 0l-2.879-2.879a1 1 0 010-1.414L7 9m4 4l2.879 2.879a1 1 0 001.414 0L17 15M10 9.414l-2.879-2.879a1 1 0 00-1.414 0L3 9.414m11 1.414l-2.879-2.879a1 1 0 00-1.414 0L9 10.828" /></svg>
);
const ChartBarIcon = (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
);
const LightBulbIcon = (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
);
const CubeTransparentIcon = (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M12 17v-2.5M12 17H9.5M12 17H14.5M17 14l-1-2m0 0l-1 2m1-2H14.5M7 14l1-2m0 0l1 2m-1-2H9.5M12 21a9 9 0 110-18 9 9 0 010 18z" /></svg>
);
const PenToolIcon = (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.5L15.232 5.232z" /></svg>
);


export const TOOLS: Tool[] = [
    // Generación y Asistencia Creativa
    { id: 'design-from-prompt', title: 'Diseño desde prompts', description: 'Generar logos, ilustraciones o composiciones completas describiendo la idea.', category: 'Generación y Asistencia Creativa', icon: SparklesIcon, isImplemented: true },
    { id: 'visual-brainstorming', title: 'Brainstorming visual', description: 'Crear múltiples variantes de un concepto automáticamente.', category: 'Generación y Asistencia Creativa', icon: SparklesIcon, isImplemented: true },
    { id: 'auto-mockups', title: 'Mockups automáticos', description: 'Generar presentaciones realistas de diseños en diferentes contextos.', category: 'Generación y Asistencia Creativa', icon: SparklesIcon, isImplemented: true },
    { id: 'smart-color-palettes', title: 'Paletas de color inteligentes', description: 'Sugerir combinaciones basadas en tendencias y teoría del color.', category: 'Generación y Asistencia Creativa', icon: SparklesIcon, isImplemented: true },

    // Optimización de Flujo de Trabajo
    { id: 'auto-tagging', title: 'Auto-tagging de recursos', description: 'Clasificar automáticamente imágenes, fuentes y recursos.', category: 'Optimización de Flujo de Trabajo', icon: AdjustmentsIcon, isImplemented: true },
    { id: 'semantic-search', title: 'Búsqueda semántica', description: 'Encontrar assets usando descripciones naturales.', category: 'Optimización de Flujo de Trabajo', icon: AdjustmentsIcon, isImplemented: false },
    { id: 'task-automation', title: 'Automación de tareas repetitivas', description: 'Redimensionar, exportar formatos, preparar archivos para impresión.', category: 'Optimización de Flujo de Trabajo', icon: AdjustmentsIcon, isImplemented: false },
    { id: 'color-compatibility', title: 'Análisis de compatibilidad de colores', description: 'Verificar accesibilidad y contraste.', category: 'Optimización de Flujo de Trabajo', icon: AdjustmentsIcon, isImplemented: true },

    // Herramientas de Edición Avanzada
    { id: 'bg-removal', title: 'Remover fondo ultra-preciso', description: 'Con un clic usando segmentación semántica.', category: 'Herramientas de Edición Avanzada', icon: ScissorsIcon, isImplemented: true },
    { id: 'image-expansion', title: 'Expansión de imágenes (Outpainting)', description: 'Rellenar áreas faltantes coherentemente.', category: 'Herramientas de Edición Avanzada', icon: ScissorsIcon, isImplemented: true },
    { id: 'resolution-upscaling', title: 'Mejora de resolución (Upscaling)', description: 'Aumentar calidad de imágenes pixeladas.', category: 'Herramientas de Edición Avanzada', icon: ScissorsIcon, isImplemented: true },
    { id: 'photo-restoration', title: 'Restauración de fotos antiguas', description: 'Eliminar rayones, mejorar detalles.', category: 'Herramientas de Edición Avanzada', icon: ScissorsIcon, isImplemented: true },
    { id: 'ai-vectorizer', title: 'Vectorizador IA', description: 'Convierte bocetos, dibujos o fotos en trazos vectoriales limpios y exportables a PDF.', category: 'Herramientas de Edición Avanzada', icon: PenToolIcon, isImplemented: true },

    // Análisis y Optimización
    { id: 'trend-analysis', title: 'Análisis de tendencias', description: 'Sugerir estilos basados en mercados objetivos.', category: 'Análisis y Optimización', icon: ChartBarIcon, isImplemented: true },
    { id: 'ab-testing', title: 'Pruebas A/B automatizadas', description: 'Generar variantes y predecir performance.', category: 'Análisis y Optimización', icon: ChartBarIcon, isImplemented: true },
    { id: 'plagiarism-detection', title: 'Detección de plagio', description: 'Verificar originalidad del diseño.', category: 'Análisis y Optimización', icon: ChartBarIcon, isImplemented: false },
    { id: 'social-media-optimization', title: 'Optimización para redes sociales', description: 'Sugerir formatos y tamaños ideales.', category: 'Análisis y Optimización', icon: ChartBarIcon, isImplemented: true },

    // Asistente de Diseño Inteligente
    { id: 'automated-critique', title: 'Crítica constructiva automatizada', description: 'Análisis de composición, balance, jerarquía visual.', category: 'Asistente de Diseño Inteligente', icon: LightBulbIcon, isImplemented: true },
    { id: 'improvement-suggestions', title: 'Sugerencias de mejora', description: 'Recomendaciones específicas para refinamiento.', category: 'Asistente de Diseño Inteligente', icon: LightBulbIcon, isImplemented: true },
    { id: 'error-detection', title: 'Detección de errores', description: 'Espacios de color incorrectos, resoluciones bajas.', category: 'Asistente de Diseño Inteligente', icon: LightBulbIcon, isImplemented: true },
    { id: 'auto-style-guides', title: 'Guías de estilo automáticas', description: 'Generar manuales de marca a partir de diseños existentes.', category: 'Asistente de Diseño Inteligente', icon: LightBulbIcon, isImplemented: true },

    // Funcionalidades Específicas por Tipo de Proyecto
    { id: 'logo-design', title: 'Diseño de logos', description: 'Generación ilimitada de conceptos + variaciones.', category: 'Funcionalidades Específicas por Proyecto', icon: CubeTransparentIcon, isImplemented: true },
    { id: 'web-ui-design', title: 'Diseño web/UI', description: 'Sugerencias de layout y componentes.', category: 'Funcionalidades Específicas por Proyecto', icon: CubeTransparentIcon, isImplemented: true },
    { id: 'packaging-design', title: 'Packaging', description: 'Análisis de impacto en estantería.', category: 'Funcionalidades Específicas por Proyecto', icon: CubeTransparentIcon, isImplemented: true },
    { id: 'typography', title: 'Tipografía', description: 'Recomendaciones de pairing de fuentes inteligente.', category: 'Funcionalidades Específicas por Proyecto', icon: CubeTransparentIcon, isImplemented: true },
];
