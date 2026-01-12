from fastapi import FastAPI
from pydantic import BaseModel
from model_services.model_service_factory import create_model_service

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/llama_test")
async def llama_test():
    from llama_cpp import Llama

    llm = Llama.from_pretrained(
        repo_id="tschomacker/lora_Llama-3.1-8B-Instruct-bnb-4bit_gguf",
        filename="unsloth.Q4_K_M.gguf",
        n_ctx=8192,
    )

    response = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": """Du bist ein Sprachassistent, der normalen deutschen Text in Leichte Sprache
            übersetzt. Halte dich strikt an die folgenden Regeln:
            1. Verwende kurze Sätze. Jeder Satz enthält nur eine Aussage. 2. Schreibe
            nur Aktivsätze im Format Subjekt-Verb-Objekt. 3. Vermeide den Konjunktiv.
            Schreibe nur in der Wirklichkeitsform. 4. Ersetze den Genitiv durch präpositionale Fügungen mit „von“. 5. Nutze keine Synonyme oder Sonderzeichen.
            6. Formuliere Verneinungen positiv, wenn möglich. 7. Verwende präzise Mengenangaben nur in Ausnahmefällen; ersetze sie durch Begriffe wie „viel“ oder
            „wenig“. Jahreszahlen werden durch Ausdrücke wie „vor langer Zeit“ ersetzt. 8.
            Leichte Sprache ist keine Kindersprache. Verwende „Du“ und „Sie“ korrekt wie
            in der Standardsprache.
            Rechtschreibregeln: 1. Verwende Bindestriche oder Mediopunkte, um Zusammensetzungen zu verdeutlichen (z. B. Welt-All oder Welt·all). 2. Nutze keine
            durchgehenden Großbuchstaben oder Kursivschrift. 3. Schreibe jeden Satz in
            eine eigene Zeile.
            Regeln für den Inhalt: 1. Vermeide abstrakte Begriffe oder erkläre sie mit
            anschaulichen Beispielen. 2. Vermeide bildhafte Sprache. 3. Erkläre Fremd- und
            Fachwörter bei der ersten Erwähnung. 4. Schreibe Abkürzungen beim ersten
            Vorkommen aus.
            Gestaltungsregeln: 1. Gestalte den Text übersichtlich. Jeder Satz steht in
            einer eigenen Zeile und linksbündig im Flattersatz. 2. Verwende Aufzählungspunkte
            für Listen. 3. Bilder und Text sollen getrennt bleiben. Bilder können zur Veranschaulichung hinzugefügt werden.
            Beispiel: Normaler Text: 'Bringen Sie für Ihre stationäre Aufnahme in eine
            unserer Kliniken bitte die notwendigen Formulare, sewie den Einweisungsschein
            Ihres Arztes, die Krankenversicherungskarte und gültige Personalpapiere mit.'
            Leichte Sprache: 'Sie sind krank?
            Sie müssen ins Kranken-Haus zu einer Untersuchung? Oder zu einer Operation?
            Dann bringen Sie bitte diese Papiere mit ins Kranken-Haus:
            Den Zettel von Ihrem Haus-Arzt. Auf diesem Zettel steht, dass Sie ins Kranken-Haus müssen. Die Karte der Kranken-Versicherung. Ihren Ausweis.
            Bitte schauen Sie vorher in Ihrem Ausweis ein Datum nach.
            Unter dem Punkt „Ablaufdatum“ steht ein Datum mit einer Jahreszahl.
            Ist das dort angegebene Datum schon vorbei?
            Dann gilt dieser Ausweis nicht mehr.
            Sie müssen dann einen anderen Ausweis mitbringen,
            z. B. den Reisepass."""},
                    {
                        "role": "user",
                        "content": "Je nach Art, Gewicht und Größe Ihres Tieres sowie den gültigen Länderbestimmungen ist der Transport gegen Gebühr entweder als zusätzliches Handgepäck in der Passagierkabine oder als Übergepäck im klimatisierten Frachtraum des Passagierflugzeugs möglich. Zusätzlich bietet Lufthansa Cargo den Transport von diesen und anderen Tierarten als Luftfracht an. Für Assistenzhunde gelten Sonderregelungen. Bitte beachten Sie, dass aufgrund von örtlichen Gegebenheiten und Einschränkungen der Transport von großen Tieren im Frachtraum zu bestimmten Zielen nicht möglich ist."
                    }
                ]
            )
    return {"response": response["choices"][0]["message"]["content"]}

class SimplifyRequest(BaseModel):
    input_text: str
    selected_service: str

async def call_service_method(input_text: str, selected_service: str, method_name: str):
    modelService = create_model_service(selected_service)
    method = getattr(modelService, method_name)
    return method(input_text)

@app.post("/generate_text")
async def simplipy(req: SimplifyRequest):
    input_text = req.input_text
    selected_service = req.selected_service

    modelService = create_model_service(selected_service)
    res = modelService.generate_simplified_text(input_text)
    return {"response": res}

@app.post("/sentence_simplifications")
async def sentence_simplifications(req: SimplifyRequest):
    return {"response": await call_service_method(req.input_text, req.selected_service, "generate_sentence_simplifications")}

@app.post("/sentence_suggestions")
async def sentence_suggestions(req: SimplifyRequest):
    return {"response": await call_service_method(req.input_text, req.selected_service, "generate_sentence_suggestions")}