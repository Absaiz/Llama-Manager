# OpenCvSharp4 — API Reference para llama-server code tests

## Namespace principal
```csharp
using OpenCvSharp;
```
No uses `OpenCvSharp.Extensions`, `CvInvoke`, `Imgproc`, `MatOfPoints2` ni nada de EmguCV.
Todas las funciones son métodos estáticos de `Cv2`.

---

## Cargar imagen desde byte[]
```csharp
Mat mat = Cv2.ImDecode(pngBytes, ImreadModes.Color);  // BGR, no RGB
```

## Conversión de color
```csharp
Mat hsv = new Mat();
Cv2.CvtColor(mat, hsv, ColorConversionCodes.BGR2HSV);  // NO RGB2HSV
```

## Umbralización HSV (máscara binaria)
```csharp
Mat mask = new Mat();
Cv2.InRange(hsv,
    new Scalar(hueMin, satMin, valMin),   // lower bound
    new Scalar(hueMax, satMax, valMax),   // upper bound
    mask);
```

## Encontrar contornos
```csharp
Point[][] contours;
HierarchyIndex[] hierarchy;
Cv2.FindContours(
    mask,
    out contours,
    out hierarchy,
    RetrievalModes.List,          // o External, Tree
    ContourApproximationModes.ApproxSimple);
```

## Área de contorno
```csharp
double area = Cv2.ContourArea(contours[i]);
```

## Dibujar contornos en máscara
```csharp
Cv2.DrawContours(filteredMask, contours, i, new Scalar(255), -1);
```

## Acceso a datos raw (unsafe)
```csharp
// Mat.Data es IntPtr
// Mat.Step es int (bytes por fila, incluye padding)
unsafe
{
    byte* ptr = (byte*)mask.Data;
    int step  = mask.Step;
    int width = mask.Width;
    int height= mask.Height;

    for (int y = 0; y < height; y++)
    {
        byte* row = ptr + y * step;
        for (int x = 0; x < width; x++)
        {
            if (row[x] > 0)
            {
                // pixel detectado en (x, y)
            }
        }
    }
}
```

## Alternativa por columnas (para perfil láser)
```csharp
unsafe
{
    byte* ptr = (byte*)mask.Data;
    int step  = mask.Step;

    for (int x = 0; x < mask.Width; x++)
    {
        long sumY = 0; int count = 0;
        byte* col = ptr + x;
        for (int y = 0; y < mask.Height; y++)
        {
            if (*col > 0) { sumY += y; count++; }
            col += step;
        }
        if (count > 0)
            avgY[x] = (int)(sumY / count);
    }
}
```

## Dispose — siempre usar `using`
```csharp
using var mat  = Cv2.ImDecode(bytes, ImreadModes.Color);
using var hsv  = new Mat();
using var mask = new Mat();
// Se liberan automáticamente al salir del bloque
```

## Habilitar unsafe en .csproj
```xml
<PropertyGroup>
  <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
</PropertyGroup>
```

## NuGet requerido
```
dotnet add package OpenCvSharp4
dotnet add package OpenCvSharp4.runtime.win   # Windows
# dotnet add package OpenCvSharp4.runtime.linux # Linux
```

## Errores comunes
| Error | Causa | Corrección |
|---|---|---|
| `CvInvoke` no existe | Confusión con EmguCV | Usa `Cv2.` |
| `VectorOfPoint` | EmguCV | Usa `Point[][]` |
| `MatOfPoints2` | EmguCV | No existe en OpenCvSharp4 |
| `RGB2HSV` | Orden incorrecto | OpenCV usa BGR, usa `BGR2HSV` |
| `Mat.data` minúscula | C++ API | En C# es `Mat.Data` (mayúscula) |
| `Mat.FromMemory()` | No existe | Usa `Cv2.ImDecode()` |
