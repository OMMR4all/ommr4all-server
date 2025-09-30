from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        # Fügen Sie die gleichen Daten hinzu wie in Ihrem jwt_response_payload_handler
        data['permissions'] = self.user.get_all_permissions()
        return data