open System
open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open FSharp.Data
open XPlot.Plotly

[<CLIMutable>]
type LyricsInput = 
    {
        Song : string
        Artist : string
        Genre : string
        Lyrics : string
        Year : int
    }

[<CLIMutable>]
type GenrePrediction = 
    {
        [<ColumnName("PredictedLabel")>]
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

[<EntryPoint>]
let main _argv =
    let appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs().[0])
    let dataModelPath = Path.Combine(appPath,"../../../","Data", "Model")
    let trainDataPath  = Path.Combine(appPath,"../../../","Data","lyrics.csv")
    let trainDataPath = Path.Combine("../","Data","lyrics.csv")
    
    
    let msft = CsvFile.Load(File.Open(trainDataPath, FileMode.Open), separators = ",", quote = '"', hasHeaders= true)
    let songLyrics = 
               msft.Rows
               |> Seq.filter (fun row -> not(row.GetColumn "lyrics" |> String.IsNullOrEmpty))
               |> Seq.filter (fun row -> not(String.Equals(row.GetColumn "lyrics", "[Instrumental]", StringComparison.OrdinalIgnoreCase)))
               |> Seq.map (fun row -> {  Song = (row.GetColumn "song")
                                         Artist = (row.GetColumn "artist")
                                         Genre = (row.GetColumn "genre")
                                         Lyrics = (row.GetColumn "lyrics").Replace(Environment.NewLine, ", ")
                                         Year = (row.GetColumn "year") |> int
                                      })
               
               
    tokenizeLyrics songLyrics
         |> renderLineChartForWords
         |> ignore
     
    Console.ReadLine() |> ignore
    0
       
       
       