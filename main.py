from fastapi import FastAPI
from fastapi.responses import JSONResponse



from retrieval_pipeline import answer_question 

app = FastAPI()


@app.post("/ask/")
async def ask_question(question: str):
    try:
        answer = answer_question(question)
        return {
            "question": question,
            "answer": answer
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
