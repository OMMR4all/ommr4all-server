from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        data['permissions'] = self.user.get_all_permissions()
        # superusers already receive every database.* permission through get_all_permissions(),
        # plain staff users do not -- see restapi.views.auth.is_admin
        data['is_admin'] = self.user.is_superuser or self.user.is_staff
        return data