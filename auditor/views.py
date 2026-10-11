from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services import detect_ppe, audit_ppe_compliance, ask_statutory_copilot

class AuditFrameView(APIView):
    """POST /api/audit/
    Accepts an uploaded image file (multipart/form-data) under key 'image'.
    Returns annotated base64 image, detected entities, and statutory audit text.
    """
    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get("image")
        if not file_obj:
            return Response({"error": "No image file provided in key 'image'."}, status=status.HTTP_400_BAD_REQUEST)

        image_bytes = file_obj.read()
        annotated_base64, detections = detect_ppe(image_bytes)
        audit_report = audit_ppe_compliance(detections)

        return Response({
            "success": True,
            "detections": list(set(detections)),
            "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
            "audit_report": audit_report,
        }, status=status.HTTP_200_OK)

class CopilotQueryView(APIView):
    """POST /api/copilot/
    Accepts JSON: {"query": "..."}
    Returns copilot response.
    """
    def post(self, request, *args, **kwargs):
        query = request.data.get("query")
        if not query:
            return Response({"error": "Query string is required."}, status=status.HTTP_400_BAD_REQUEST)

        answer = ask_statutory_copilot(query)
        return Response({"success": True, "answer": answer}, status=status.HTTP_200_OK)