using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Globalization;

namespace myApp
{
    class Program
    {
        // Шаг 1: Определите ваши структуры данных
        // IrisData используется для предоставления обучающих данных, а также
        // как введение для предиктивных операций
        // - Первые 4 свойства -- это входные данные / функции, используемые для прогнозирования метки label
        // - Label -- это то, что вы предсказываете, и устанавливается только при обучении
        public class IrisData
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string Label;
        }

        // IrisPrediction является результатом, возвращенным из операций прогнозирования
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        static void Main(string[] args)
        {
            // Шаг 2: создание среды ML.NET 
            var mlContext = new MLContext();

            // Если работаете в Visual Studio, убедитесь, что параметр 'Copy to Output Directory'
            // iris-data.txt установлен как 'Copy always'
            var reader = mlContext.Data.CreateTextReader<IrisData>(separatorChar: ',', hasHeader: true);
            Microsoft.ML.Data.IDataView trainingDataView = reader.Read("iris.data");

            // Шаг 3: Преобразуйте свои данные и добавьте learner
            // Присвойте числовые значения тексту в столбце «label», потому что только
            // числа могут быть обработаны во время обучения модели.
            // Добавьте обучающий алгоритм в pipeline. Например (What type of iris is this?)
            // Преобразовать label обратно в исходный текст (после преобразования в число на шаге 3)
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Шаг 4: обучите модель на этом дата-сете
            var model = pipeline.Fit(trainingDataView);

            // Шаг 5: используйте модель для предсказания

            string input = string.Empty;
            string exit = String.Empty;
            while (exit != "x")
            {
                Console.Clear();
                Console.WriteLine("Enter data:");
                input = Console.ReadLine();
                string[] arr = input.Split(',');

                float a = float.Parse(arr[0], CultureInfo.InvariantCulture.NumberFormat);
                float b = float.Parse(arr[1], CultureInfo.InvariantCulture.NumberFormat);
                float c = float.Parse(arr[2], CultureInfo.InvariantCulture.NumberFormat);
                float d = float.Parse(arr[3], CultureInfo.InvariantCulture.NumberFormat);

                var prediction = model.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext).Predict(
                    new IrisData()
                    {
                        SepalLength = a,
                        SepalWidth = b,
                        PetalLength = c,
                        PetalWidth = d,
                    });

                Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
                Console.WriteLine($"Press \"x\" to exit, any other for continue.");
                exit = Console.ReadLine();
            }
        }
    }
}