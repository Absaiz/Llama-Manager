# Blazor Server — API Reference para llama-server code tests

## Namespace principal
```csharp
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using Microsoft.JSInterop;
```

---

## Subida de ficheros — InputFile (CORRECTO)
```csharp
// En el .razor
<InputFile OnChange="HandleFile" accept="image/png,image/jpeg" />

@code {
    private async Task HandleFile(InputFileChangeEventArgs e)
    {
        IBrowserFile file = e.File;
        
        // Leer como byte[]
        using var stream = file.OpenReadStream(maxAllowedSize: 10 * 1024 * 1024); // 10MB
        using var ms = new MemoryStream();
        await stream.CopyToAsync(ms);
        byte[] bytes = ms.ToArray();
    }
}
```

## NUNCA uses esto (no existe en Blazor Server):
```
// INCORRECTO:
file.OpenReader()           // no existe
file.ReadBytesAsync()       // no existe
@pageModel                  // directiva inventada
document.getElementById()   // es JavaScript, no C#
JSFunctionRef               // no existe
```

---

## ComponentBase — ciclo de vida correcto
```csharp
public class MiComponenteBase : ComponentBase
{
    // CORRECTO: bool firstRender es obligatorio
    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (firstRender)
        {
            // Solo se ejecuta una vez
        }
        await base.OnAfterRenderAsync(firstRender);
    }

    // CORRECTO: OnInitializedAsync para carga inicial
    protected override async Task OnInitializedAsync()
    {
        await base.OnInitializedAsync();
    }

    // CORRECTO: IDisposable
    public void Dispose()
    {
        // limpieza
    }
}
```

---

## Inyección de dependencias en Razor
```razor
@inject ILaserDetectionService LaserService
@inject IJSRuntime JS
@inject IConfiguration Config
@inject ILogger<MiComponente> Logger
```

En code-behind (clase separada):
```csharp
[Inject] public ILaserDetectionService LaserService { get; set; } = null!;
[Inject] public IJSRuntime JS { get; set; } = null!;
```

---

## IJSRuntime — interop con JavaScript
```csharp
// Llamar función JS desde C#
await JS.InvokeVoidAsync("console.log", "hola");
var resultado = await JS.InvokeAsync<string>("miFuncion", arg1, arg2);

// NO existe:
// JS.InvokeAsync<IJSObjectReference>("addEventListener", ...)  ← incorrecto
// new JSFunctionRef(...)  ← no existe
```

---

## IConfiguration — lectura de appsettings.json
```csharp
// CORRECTO: GetValue<T> devuelve T, no T?
// Para double, usa el segundo parámetro como default:
double hueMin = config.GetValue<double>("Laser:HsvMinHue", 45.0);

// INCORRECTO (no compila):
// double hueMin = config.GetValue<double>("Laser:HsvMinHue") ?? 45;  ← double no es nullable
```

---

## Program.cs — Blazor Server .NET 8 correcto
```csharp
var builder = WebApplication.CreateBuilder(args);

// Servicios Blazor Server
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// Tus servicios
builder.Services.AddScoped<ILaserDetectionService, LaserDetectionService>();
// O Singleton si el servicio es thread-safe:
builder.Services.AddSingleton<ILaserDetectionService, LaserDetectionService>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
   .AddInteractiveServerRenderMode();

app.Run();
```

---

## Componente Razor completo y correcto
```razor
@page "/laser"
@inject ILaserDetectionService LaserService

<h3>Laser Profilometer</h3>

<InputFile OnChange="HandleFile" accept="image/png" />

@if (_processing)
{
    <p>Procesando...</p>
}

@if (_profile != null)
{
    <p>Puntos detectados: @_profile.Points.Count</p>
    <p>Zero Reference Y: @_profile.ZeroReferenceY</p>
    <p>Altura máxima: @(_profile.Heights.Any() ? _profile.Heights.Max() : 0):F2</p>
}

@code {
    private bool _processing;
    private LaserProfile? _profile;

    private async Task HandleFile(InputFileChangeEventArgs e)
    {
        _processing = true;
        _profile = null;

        try
        {
            using var stream = e.File.OpenReadStream(maxAllowedSize: 10 * 1024 * 1024);
            using var ms = new MemoryStream();
            await stream.CopyToAsync(ms);

            _profile = await LaserService.DetectAsync(ms.ToArray());
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
        finally
        {
            _processing = false;
        }
    }
}
```

---

## App.razor correcto (.NET 8)
```razor
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8" />
    <HeadOutlet />
</head>
<body>
    <Routes />
    <script src="_framework/blazor.web.js"></script>
</body>
</html>
```

---

## .csproj correcto — Blazor Server .NET 8
```xml
<Project Sdk="Microsoft.NET.Sdk.Web">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="OpenCvSharp4" Version="4.10.0.20240616" />
    <PackageReference Include="OpenCvSharp4.runtime.win" Version="4.10.0.20240616" />
  </ItemGroup>
</Project>
```

---

## Errores comunes
| Error | Causa | Corrección |
|---|---|---|
| `@pageModel` | Directiva inventada | No existe, usa `@inherits` o `@inject` |
| `file.OpenReader()` | No existe en IBrowserFile | Usa `file.OpenReadStream()` |
| `GetValue<double>() ?? 45` | double no es nullable | Usa `GetValue<double>("key", 45.0)` |
| `OnAfterRenderAsync()` sin bool | Firma incorrecta | `OnAfterRenderAsync(bool firstRender)` |
| `Microsoft.JavascriptInterop` | Namespace inventado | `Microsoft.JSInterop` |
| `JSFunctionRef` | No existe | Usa `DotNetObjectReference` o JS puro |
| `base.Dispose()` | ComponentBase no tiene Dispose | Implementa `IDisposable` explícitamente |
| `AddRazorComponents()` sin `.AddInteractiveServerComponents()` | Incompleto en .NET 8 | Añadir el chain |
