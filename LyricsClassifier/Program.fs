open System
open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open FSharp.Data
open XPlot.Plotly

[<CLIMutable>]
type LyricsInput = 
    {
        Index: int 
        Song : string
        Artist : string
        Genre : string
        Lyrics : string
        Year : int
    }

[<CLIMutable>]
type GenrePrediction = 
    {
        [<ColumnName("Label")>]
        Genre : string
        Score : float32 []
    }

let downcastPipeline (x : IEstimator<_>) = 
    match x with 
    | :? IEstimator<ITransformer> as y -> y
    | _ -> failwith "downcastPipeline: expecting a IEstimator<ITransformer>"
    
let stopwords = [|"ourselves"; "hers"; "between"; "yourself"; "but"; "again"; "there"; "about"; "once"; "during"; "out"; "very"; "having"; "with"; "they"; "own"; "an"; "be"; "some"; "for"; "do"; "its"; "yours"; "such"; "into"; "of"; "most"; "itself"; "other"; "off"; "is"; "s"; "am"; "or"; "who"; "as"; "from"; "him"; "each"; "the"; "themselves"; "until"; "below"; "are"; "we"; "these"; "your"; "his"; "through"; "don"; "nor"; "me"; "were"; "her"; "more"; "himself"; "this"; "down"; "should"; "our"; "their"; "while"; "above"; "both"; "up"; "to"; "ours"; "had"; "she"; "all"; "no"; "when"; "at"; "any"; "before"; "them"; "same"; "and"; "been"; "have"; "in"; "will"; "on"; "does"; "yourselves"; "then"; "that"; "because"; "what"; "over"; "why"; "so"; "can"; "did"; "not"; "now"; "under"; "he"; "you"; "herself"; "has"; "just"; "where"; "too"; "only"; "myself"; "which"; "those"; "i"; "after"; "few"; "whom"; "t";"ll"; "being"; "if"; "theirs"; "my"; "against"; "a"; "by"; "doing"; "it"; "how"; "further"; "was"; "here"; "than"; "s"; "t"; "m"; "'re"; "'ll";"ve";"..."; "ä±"; "''"; "``"; "--"; "'d"; "el"; "la"; "que"; "y"; "de"; "en"|]
let symbols = [|'\''; ' '; ','|]
            
let renderLineChartForWords(words: seq<string>) = 
            words
                |> Seq.countBy id 
                |> Seq.sortByDescending(fun (value:string, count :int) -> count)
                |> Seq.take 15
                |> Chart.Line
                        
let tokenizeLyrics (lyrics: seq<LyricsInput>) =
            let mlContext = MLContext(seed = Nullable 0)        
            let data = mlContext.Data.LoadFromEnumerable lyrics
            
            let pipeline = mlContext.Transforms.Text.FeaturizeText("FeaturizedLyrics", "Lyrics")
                                            .Append(mlContext.Transforms.Text.NormalizeText("NormalizedLyrics", "Lyrics"))
                                            .Append(mlContext.Transforms.Text.TokenizeWords("TokenizedLyric", "NormalizedLyrics", symbols))
                                            .Append(mlContext.Transforms.Text.RemoveStopWords("LyricsWithNoCustomStopWords", "TokenizedLyric", stopwords))
                                            .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("LyricsWithNoStopWords", "LyricsWithNoCustomStopWords"))

            let transformedData = pipeline.Fit(data).Transform(data)
            transformedData.GetColumn<string[]>(mlContext, "LyricsWithNoStopWords")
                        |> Seq.concat
                        |> Seq.toList

let buildAndTrainModel (lyrics: seq<LyricsInput>) =

    // Create MLContext to be shared across the model creation workflow objects 
    // Set a random seed for repeatable/deterministic results across multiple trainings.
    let mlContext = MLContext(seed = Nullable 0)

    // STEP 1: Common data loading configuration
    let textLoader =
        mlContext.Data.LoadFromEnumerable<LyricsInput>(lyrics)
    let trainingDataView = mlContext.Data.LoadFromEnumerable<LyricsInput>(lyrics)
       
    // STEP 2: Common data process configuration with pipeline data transformations
    let dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("FeaturizedLyrics", "Lyrics")
                                                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Genre"))
                                                .Append(mlContext.Transforms.Text.NormalizeText("NormalizedLyrics", "Lyrics"))
                                                .Append(mlContext.Transforms.Text.TokenizeWords("TokenizedLyric", "NormalizedLyrics", symbols))
                                                .Append(mlContext.Transforms.Text.RemoveStopWords("LyricsWithNoCustomStopWords", "TokenizedLyric", stopwords))
                                                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("LyricsWithNoStopWords", "LyricsWithNoCustomStopWords"))
                                                .Append(mlContext.Transforms.Text.FeaturizeText("LyricsFeaturized", "LyricsWithNoStopWords"))
                                                .Append(mlContext.Transforms.Text.FeaturizeText("SongFeaturized", "Song"))
                                                .Append(mlContext.Transforms.Text.FeaturizeText("ArtistFeaturized", "Artist"))
                                                .Append(mlContext.Transforms.Concatenate("Features", "FeaturizedLyrics"))
                                                |> downcastPipeline

    // (OPTIONAL) Peek data (such as 2 records) in training DataView after applying the ProcessPipeline's transformations into "Features" 
    Common.ConsoleHelper.peekDataViewInConsole<LyricsInput> mlContext trainingDataView dataProcessPipeline 2 |> ignore
    Common.ConsoleHelper.peekVectorColumnDataInConsole mlContext "Features" trainingDataView dataProcessPipeline 2 |> ignore
    
    // STEP 3: Create the selected training algorithm/trainer
    let trainer =
       mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent( "Label", "Features") |> downcastPipeline

    //Set the trainer/algorithm
    let modelBuilder = 
        dataProcessPipeline
            .Append(trainer)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
    
    // STEP 4: Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
    // in order to evaluate and get the model's accuracy metrics
    printfn "=============== Cross-validating to get model's accuracy metrics ==============="

    //Measure cross-validation time
    let watchCrossValTime = System.Diagnostics.Stopwatch.StartNew()

    trainingDataView.Preview() |> ignore;

    //Stop measuring time
    watchCrossValTime.Stop()
    printfn "Time Cross-Validating: %d miliSecs"  watchCrossValTime.ElapsedMilliseconds
 
    // STEP 5: Train the model fitting to the DataSet
    printfn "=============== Training the model ==============="
    let trainedModel = modelBuilder.Fit(trainingDataView)


    // (OPTIONAL) Try/test a single prediction with the "just-trained model" (Before saving the model)
    let issue = {
        Index = 0
        Song = ""
        Artist =""
        Genre =""
        Lyrics = ""
        Year = 1994
    }
    let predEngine = mlContext.Model.CreatePredictionEngine<LyricsInput, GenrePrediction>(trainedModel)
    let prediction =  predEngine.Predict(issue)

    printfn "=============== Single Prediction just-trained-model - Result: %s ===============" prediction.Genre

    // STEP 6: Save/persist the trained model to a .ZIP file
    printfn "=============== Saving the model to a file ==============="
    do 
        use f = File.Open("",FileMode.Create)
        mlContext.Model.Save(trainedModel, f)

    Common.ConsoleHelper.consoleWriteHeader "Training process finalized"

[<EntryPoint>]
let main _argv =
    let appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs().[0])
    let dataModelPath = Path.Combine(appPath,"../../../","Data", "Model")
    let trainDataPath  = Path.Combine(appPath,"../../../","Data","lyrics.csv")
    let trainDataPath = Path.Combine("../","Data","lyrics.csv")
    
    
    let msft = CsvFile.Load(File.Open(trainDataPath, FileMode.Open), separators = ",", quote = '"', hasHeaders= true,ignoreErrors = true)
    let songLyrics = 
               msft.Rows
               |> Seq.filter (fun row -> not(row.GetColumn "lyrics" |> String.IsNullOrEmpty))
               |> Seq.filter (fun row -> not(String.Equals(row.GetColumn "lyrics", "[Instrumental]", StringComparison.OrdinalIgnoreCase)))
               |> Seq.map (fun row -> {  Index = (row.GetColumn "index")  |> int
                                         Song = (row.GetColumn "song")
                                         Artist = (row.GetColumn "artist")
                                         Genre = (row.GetColumn "genre")
                                         Lyrics = (row.GetColumn "lyrics").Replace(Environment.NewLine, ", ")
                                         Year = (row.GetColumn "year") |> int
                                      })
               |> Seq.take 10
               
    tokenizeLyrics songLyrics
         |> renderLineChartForWords
         |> ignore
     
    buildAndTrainModel songLyrics
        |> ignore
    Console.ReadLine() |> ignore
    0
       
       
       