// FIX: Import React to use the React.ReactElement type for the tool icon.
import React from 'react';

// Using a const object instead of an enum for better JS interoperability,
// which can sometimes cause issues with module loaders.
export const ToolCategory = {
    CREATIVE_ASSISTANCE: 'Generación y Asistencia Creativa',
    WORKFLOW_OPTIMIZATION: 'Optimización de Flujo de Trabajo',
    ADVANCED_EDITING: 'Herramientas de Edición Avanzada',
    ANALYSIS_OPTIMIZATION: 'Análisis y Optimización',
    DESIGN_ASSISTANT: 'Asistente de Diseño Inteligente',
    PROJECT_SPECIFIC: 'Funcionalidades Específicas por Proyecto',
} as const;

// Create a type from the object values.
export type ToolCategoryValue = (typeof ToolCategory)[keyof typeof ToolCategory];


export interface Tool {
    id: string;
    title: string;
    description: string;
    category: ToolCategoryValue; // Use the value type
    icon: React.ReactElement;
    isImplemented?: boolean;
}