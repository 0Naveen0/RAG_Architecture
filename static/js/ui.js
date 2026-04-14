
const spinner = document.getElementById("loading");
const answerBox = document.getElementById("answer");
const confidenceBox = document.getElementById("confidence");
const sourceBox = document.getElementById("source");


export function showLoading(){
    spinner.style.display = "block";
    
}

export function hideLoading(){
    spinner.style.display = "none";
}

export function displayAnswer(result){
    answerBox.textContent=result.answer;
    confidenceBox.textContent=result.confidence;
    if(result.source){
       sourceBox.textContent= `Source:${result.source} (Chunk ${result.chunk_id})`;

    }else{
        sourceBox.textContent= "";
    }
    
}

