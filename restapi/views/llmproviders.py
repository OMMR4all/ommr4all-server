from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView


class LLMProvidersView(APIView):
    """Reports which LLM text transcription providers are usable on this
    server (packages installed / API keys configured via environment
    variables). The client uses this to enable/disable the provider
    selection of the text_llm algorithm."""
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        from omr.steps.text.llm.adapters import available_providers
        return Response({'providers': available_providers()})
