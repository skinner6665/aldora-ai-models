"""
Pill Identifier — Claude claude-sonnet-4-6 Vision API
Identifica comprimidos, cápsulas e medicamentos por imagem.
CFM 2.454/2026: apoio diagnóstico, confirmar com farmacêutico antes de qualquer uso.
"""
import os
import base64
import json

SYSTEM_PROMPT = """Você é um sistema de identificação de medicamentos por imagem para uso médico hospitalar.
Analise a imagem do comprimido, cápsula ou embalagem fornecida e retorne APENAS um objeto JSON com os campos:
{
  "nome_comercial": "nome comercial mais provável",
  "nome_generico": "princípio ativo (DCI)",
  "forma": "forma farmacêutica (comprimido / cápsula / drágea / solução / etc.)",
  "cor": "cor ou cores predominantes",
  "formato": "formato geométrico (redondo / oval / oblongo / biconvexo / etc.)",
  "imprint": "código, número ou texto impresso no medicamento (ou 'não visível')",
  "uso_terapeutico": "indicação terapêutica principal",
  "confianca": "alta | media | baixa"
}
Critérios de confiança:
- alta: imagem clara, imprint legível, identificação unívoca
- media: imagem razoável ou imprint parcialmente visível
- baixa: imagem turva, múltiplas possibilidades ou ausência de características distintivas
NUNCA invente medicamentos. Se incerto, use confianca "baixa" com descrição das características visuais observadas.
Retorne SOMENTE o JSON, sem markdown, sem explicações."""


class PillModel:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def predict(self, img_bytes: bytes, mime: str = "image/jpeg") -> dict:
        import anthropic

        b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Identifique este medicamento e retorne o JSON.",
                        },
                    ],
                }
            ],
        )

        raw = message.content[0].text.strip()

        # Remove markdown code fences se presentes
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "nome_comercial": "Não identificado",
                "nome_generico": "Não identificado",
                "forma": "desconhecida",
                "cor": "desconhecida",
                "formato": "desconhecido",
                "imprint": "não visível",
                "uso_terapeutico": "desconhecido",
                "confianca": "baixa",
            }

        result["aviso"] = (
            "CFM 2.454/2026: identificação por IA como apoio diagnóstico (SaMD/ANVISA). "
            "CONFIRMAR com farmacêutico ou bula antes de qualquer administração. "
            "Não comunicar ao paciente sem mediação de profissional habilitado."
        )
        return result


def load_pill() -> PillModel:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY não configurada. "
            "Defina a variável de ambiente no painel do Railway."
        )
    return PillModel(api_key)
