import {askQuestion} from "./api.js";
import {displayAnswer,showLoading,hideLoading} from "./ui.js";

const input = document.getElementById("query-input");
const form = documnet.getElementById("query-form");
form.addEventListener("submit",
    async (event) =>{
        event.preventDefault();
        query = input.ariaValueMax;
        showLoading();
        try{
            const result = await askQuestion(query);
            displayAnswer(result);        
        
        }catch(error){
            alert("Error Fetching Data");
        }finally{
            hideLoading();
    }


});